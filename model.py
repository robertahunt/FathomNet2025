import os
import itertools
import numpy as np
import pandas as pd

from ete3 import Tree

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv
from torchvision import datasets, transforms, models

import pytorch_lightning as pl

from batchminer import BatchMiner
from utils import save_image_grid
from utils import get_T_matrix, tree_to_distance_matrix, rand_bbox, sample_triplets
from utils import run_inference, save_predictions



class GraphClassifier(torch.nn.Module):
    def __init__(self, in_channels=1280, graph_channels=128, num_classes=79):
        super().__init__()
        self.gcn1 = GCNConv(in_channels, graph_channels)
        
        self.hidden = torch.nn.Linear(in_channels + graph_channels, 512)
        self.classifier = torch.nn.Linear(512, num_classes)

        #self.hidden = torch.nn.Linear(in_channels, 512)
        #self.classifier = torch.nn.Linear(512, num_classes)

        #self.classifier = torch.nn.Linear(in_channels + graph_channels, num_classes)

    def forward(self, x, edge_index, img_sizes):
        x1 = F.relu(self.gcn1(x, edge_index))
        x = F.relu(self.hidden(torch.hstack((x,x1))))#,img_sizes.unsqueeze(1).float()))))
        #x = F.relu(self.hidden(x))
        return x, self.classifier(x)#torch.hstack((x,x1)))

class EfficientNetClassifier(pl.LightningModule):
    def __init__(self, tree_path, labels, num_classes=10, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        # Load EfficientNet B0
        self.model = models.efficientnet_b0(pretrained=True)
        # Replace classifier head
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
        self.model.graph_classifier = GraphClassifier()
        self.tree = Tree(tree_path, format=3)
        self.unique_labels = sorted(np.unique(labels))
        self.class_to_idx = {cl:idx for idx, cl in enumerate(self.unique_labels)}
        self.idx_to_class = {idx:cl for idx, cl in enumerate(self.unique_labels)}
        self.dist_matrix = torch.Tensor(tree_to_distance_matrix(tree_path, self.unique_labels).values).cuda()
        self.T = get_T_matrix(self.unique_labels, "", self.tree)
        self.sub_classes = self.get_sub_classes()
        self.ancestors = self.get_ancestors()
        self.focus_classes = ['Munidopsis', 'Gastropoda', 'Acanthoptilum', 'Benthocodon pedunculata','Sebastes', 'Asteroidea', 'Actinopterygii', 'Metridium farcimen',
       'Porifera', 'Scotoplanes globosa', 'Scleractinia','Gersemia juliepackardae', 'Elpidia', 'Keratoisis', 'Caridea']
        self.focus_classes = []#[self.class_to_idx[x] for x in self.focus_classes]

        self.class_weights = []
        for cls in range(79):
            if cls in self.focus_classes:
                self.class_weights += [2]
            else:
                self.class_weights += [1]
        self.class_weights = torch.tensor(self.class_weights).float().cuda()

        self.best_val_acc = 0
        self.triplet_batch_miner = BatchMiner()

    def forward(self, x, image_ids, img_sizes):
        x = self.model.avgpool(self.model.features(x))
        x = self.model.classifier[0](x) # dropout
        
        val_to_indices = {_id.item(): torch.where(image_ids == _id)[0].cpu().numpy().tolist() for _id in image_ids}
        edge_indices = []
        for k in val_to_indices.keys():
            edges = list(itertools.product(val_to_indices[k],val_to_indices[k]))
            edge_indices += edges
        return self.model.graph_classifier(x.squeeze().cuda(), torch.tensor(edge_indices).T.cuda(), torch.tensor(img_sizes).cuda())
        #return x, self.model.classifier(x.cuda())

    def get_sub_classes(self):
        # get all sub classes of a certain class (including the class itself)
        class_to_leaves = {}
        for node in self.tree.traverse():
            #if node.name not in self.unique_labels:
            if node.dist == 0:
                continue
            if node.is_root() or node.is_leaf():
                continue
            if node.name == 'Animalia':
                continue
            class_to_leaves[node.name] = []

            for child in node.traverse():
                if child.name in self.unique_labels:
                    class_to_leaves[node.name] += [child.name]
            

            if len(class_to_leaves[node.name]) == 1:
                del class_to_leaves[node.name]
                continue 
            if node.name not in self.class_to_idx.keys():
                self.class_to_idx[node.name] = len(self.class_to_idx.keys())
        sub_classes_idx = {self.class_to_idx[_class]:torch.tensor([self.class_to_idx[x] for x in class_to_leaves[_class]]).cuda() for _class in class_to_leaves}
        #for j in range(len(self.unique_labels)):
        #    sub_classes_idx[j] = sub_classes_idx.get(j, [])
        #sub_classes_idx = np.array([sub_classes_idx[i] for i in range(len(sub_classes_idx)) ], dtype=list)
        return sub_classes_idx
    
    def get_ancestors(self):
        ancestors = {}

        for node in self.tree.traverse():
            if node.name not in self.unique_labels:
                continue
            ancestors[node.name] = []

            for ancestor in node.get_ancestors():
                #if ancestor.name in self.unique_labels:
                if ancestor.dist > 0:
                    ancestors[node.name] += [ancestor.name]
        ancestors_idx = {self.class_to_idx[_class]:[self.class_to_idx.get(x, -1) for x in ancestors[_class]] for _class in ancestors}
        ancestors_idx = {k:[x for x in v if x >= 0] for k,v in ancestors_idx.items()}
        ancestors_idx = np.array([ancestors_idx[i] for i in range(len(ancestors_idx))], dtype=list)
        return ancestors_idx

            
    def on_validation_epoch_end(self):
        predictions, probs = run_inference(self, image_dir= 'data_test/rois', class_names=self.unique_labels)
        df = pd.DataFrame(predictions)
        df.sort_values('annotation_id').to_csv(os.path.join(self.logger.log_dir, f'{self.current_epoch}_submission_data.csv'))
        df[['annotation_id','concept_name']].sort_values('annotation_id').to_csv(os.path.join(self.logger.log_dir, f'{self.current_epoch}_submission.csv'), index=False)
        save_predictions(self.val_logits, self.val_y, self.val_paths, self.unique_labels, self.dist_matrix, os.path.join(self.logger.log_dir,f'val_predictions_epoch_{self.current_epoch}.csv'))

        val_acc  = self.trainer.callback_metrics.get("val_acc")
        if val_acc > self.best_val_acc:
            df[['annotation_id','concept_name']].sort_values('annotation_id').to_csv(os.path.join(self.logger.log_dir, f'best_{self.current_epoch}_submission.csv'), index=False)
            probs.sort_index().to_csv(os.path.join(self.logger.log_dir, f'best_{self.current_epoch}_probabilities.csv'), index=True)


    
    def training_step(self, batch, batch_idx):
        # Fancy Aug
        x, y, image_id, img_sizes = batch['x'], batch['y'], batch['image_id'], batch['img_size']

        if (batch_idx == 0):
            save_image_grid(x, os.path.join(self.logger.log_dir,'example_inputs.png'))
        # Choose augmentation type randomly: 0 = mixup, 1 = cutmix, 2 = none
        aug_type = np.random.choice([0, 1, 2], p=[0.1,0.1,0.8])

        if aug_type == 0:  # Mixup
            lam = np.random.beta(0.4, 0.4)
            index = torch.randperm(x.size(0))
            x = lam * x + (1 - lam) * x[index]
            y_a, y_b = y, y[index]

        elif aug_type == 1:  # CutMix
            lam = np.random.beta(1.0, 1.0)
            index = torch.randperm(x.size(0))
            y_a, y_b = y, y[index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
            x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))

        triplet_loss = torch.tensor(0).float().cuda()
        if aug_type in [0, 1]:
            emb, logits = self(x, image_id, img_sizes)
            loss = lam * F.cross_entropy(logits, y_a, weight=self.class_weights) + (1 - lam) * F.cross_entropy(logits, y_b, weight=self.class_weights)
            EM_tree_loss = lam * self.EM_tree_loss(logits, y_a) + (1-lam) * self.EM_tree_loss(logits, y_b)
            tree_loss = lam * self.tree_loss_function(logits, y_a) + (1-lam) * self.tree_loss_function(logits, y_b)
            MP_tree_loss = lam * self.MP_tree_loss(logits, y_a) + (1-lam) * self.MP_tree_loss(logits, y_b)
            LPL_loss = lam * self.LPL_loss(logits, y_a) + (1-lam) * self.LPL_loss(logits, y_b)
            acc = lam * (logits.argmax(dim=1) == y_a).float().mean() + (1-lam) * (logits.argmax(dim=1) == y_b).float().mean()
            # Not including triplet loss as difficult to define how it should work in this context

        else:
            emb, logits = self(x, image_id, img_sizes)
            loss = F.cross_entropy(logits, y, weight=self.class_weights)
            EM_tree_loss = self.EM_tree_loss(logits, y)
            tree_loss = self.tree_loss_function(logits, y)
            MP_tree_loss = self.MP_tree_loss(logits, y)
            LPL_loss = self.LPL_loss(logits, y)
            if len(y.unique()) > 1:
                anchors, pos, neg, weights = sample_triplets(emb, y, self.dist_matrix) #self.triplet_batch_miner(emb, y, self.dist_matrix)
                for anc, p, n, w in zip(anchors, pos, neg, weights):
                    triplet_loss += F.triplet_margin_loss(anc, p, n, w)

            acc = (logits.argmax(dim=1) == y).float().mean()

        #anchors, pos, neg, weights = sample_triplets(emb, y, self.dist_matrix)
        


        #cov_loss = self.cov_loss(emb, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        self.log("train_EM_tree_loss", EM_tree_loss, on_step=False, on_epoch=True)
        self.log("train_MP_tree_loss", MP_tree_loss, on_step=False, on_epoch=True)
        self.log("train_tree_loss", tree_loss, on_step=False, on_epoch=True)
        self.log("train_LPL_loss", LPL_loss, on_step=False, on_epoch=True)
        self.log("train_triplet_loss", triplet_loss, on_step=False, on_epoch=True)
        #self.log("train_cov_loss", cov_loss, on_step=False, on_epoch=True)
        return loss + LPL_loss + 0.01*triplet_loss

    def validation_step(self, batch, batch_idx):
        x, y, image_id, paths, img_sizes = batch['x'], batch['y'], batch['image_id'], batch['path'], batch['img_size']

        emb, logits = self(x, image_id, img_sizes)


        if (batch_idx == 0):
            self.val_logits = logits
            self.val_y = y
            self.val_paths = paths
            save_image_grid(x, os.path.join(self.logger.log_dir,'example_val_inputs.png'))
        else:
            self.val_logits = torch.vstack((self.val_logits, logits))
            self.val_y = torch.hstack((self.val_y, y))
            self.val_paths += paths
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        EM_tree_loss = self.EM_tree_loss(logits, y)
        MP_tree_loss = self.MP_tree_loss(logits, y)
        tree_loss = self.tree_loss_function(logits, y)
        LPL_loss = self.LPL_loss(logits, y)
        #anchors, pos, neg, weights = sample_triplets(emb, y, self.dist_matrix)

        triplet_loss = torch.tensor(0).float().cuda()
        if len(y.unique()) > 1:
            anchors, pos, neg, weights = sample_triplets(emb, y, self.dist_matrix)#self.triplet_batch_miner(emb, y, self.dist_matrix)
            for anc, p, n, w in zip(anchors, pos, neg, weights):
                triplet_loss += F.triplet_margin_loss(anc, p, n, w)
        #cov_loss = self.cov_loss(emb, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True)
        self.log("val_EM_tree_loss", EM_tree_loss, on_step=False, on_epoch=True)
        self.log("val_MP_tree_loss", MP_tree_loss, on_step=False, on_epoch=True)
        self.log("val_tree_loss", tree_loss, on_step=False, on_epoch=True)
        self.log("val_LPL_loss", LPL_loss, on_step=False, on_epoch=True)
        self.log("val_triplet_loss", triplet_loss, on_step=False, on_epoch=True)
        #self.log("val_cov_loss", cov_loss, on_step=False, on_epoch=True)




    def LPL_loss(self, logits, y):
        loss = 0
        n_levels = len(self.sub_classes.keys())
        for level in self.sub_classes.keys():
            sub_classes = self.sub_classes[level]
            other_classes = torch.tensor([x for x in range(79) if x not in self.sub_classes]).cuda()
            sub_logits = logits[:,sub_classes].sum(axis=1)
            other_logits = logits[:,other_classes].sum(axis=1)
            level_logits = torch.stack((other_logits, sub_logits)).T
            level_y = torch.isin(y,sub_classes).long()
            loss += F.cross_entropy(level_logits, level_y)
        return 2*loss/n_levels

    def cov_loss(self, batch, classes):
        indices = classes.argsort()
        classes = classes[indices]
        classes = [self.idx_to_class[x.item()] for x in classes]
        z = batch[indices]


        sub_cov = 0.01*torch.tensor(self.T.loc[classes, classes].values).cuda()
        est_cov = torch.mean(z.T[:, None, :] * z.T[:, :, None], axis=0)

        return torch.mean((sub_cov - est_cov)**2)

    def EM_tree_loss(self, logits, y):
        # tree loss if we choose the one with the lowest expected loss given the probabilities and the distance matrix
        best_choice = (torch.softmax(logits, 1) @ self.dist_matrix).argmin(1)
        return self.dist_matrix[best_choice,y].mean()

    def MP_tree_loss(self, logits, y):
        # tree loss if we choose the one with the maximum overall probabillity / likelihood
        return self.dist_matrix[logits.argmax(dim=1),y].mean()

    def tree_loss_function(self, logits, y):
        return (torch.softmax(logits, 1) * self.dist_matrix[y]).sum(axis=1).mean()

    #def tree_gaussian_similarity_function(self, logits, y):
    #    return (torch.softmax(logits, 1) * self.dist_matrix[y]).sum(axis=1).mean()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
