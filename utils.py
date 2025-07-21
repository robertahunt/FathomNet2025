import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.utils as vutils

from ete3 import Tree
from tqdm import tqdm

import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

from sampler import GroupedPrefixSampler
from dataset import InferenceDataset


def sample_triplets(embeddings: torch.Tensor, labels: torch.Tensor, dist_matrix: torch.tensor):
    """
    Samples triplets (anchor, positive, negative) from a batch of embeddings.

    Args:
        embeddings (torch.Tensor): Tensor of shape (batch_size, embedding_dim)
        labels (torch.Tensor): Tensor of shape (batch_size,) with class labels

    Returns:
        List of (anchor, positive, negative) triplet tuples (each is a torch.Tensor)
    """
    anchors = []
    positives = []
    negatives = []
    weights = []

    for anchor_idx in range(len(embeddings)):
        anchor_label = labels[anchor_idx]

        # Get indices of other samples with same label
        positive_indices = [i for i, l in enumerate(labels) if l == anchor_label and i != anchor_idx]
        if not positive_indices:
            continue  # skip if no other positive sample

        # Get indices of samples with different label
        negative_indices = [i for i, l in enumerate(labels) if l != anchor_label]
        if not negative_indices:
            continue  # skip if no negative sample
        # Randomly select positive and negative
        positive_idx = random.choice(positive_indices)
        negative_idx = random.choice(negative_indices)


        weight = dist_matrix[anchor_label, labels[negative_idx]].item()
        if weight == 0:
            continue

        anchors += [embeddings[anchor_idx]]
        positives += [embeddings[positive_idx]]
        negatives += [embeddings[negative_idx]]
        weights += [weight]

    return anchors, positives, negatives, torch.tensor(weights).cuda()

def save_predictions(logits, ys, paths, class_names, dist_matrix, save_path):
    probs = torch.softmax(logits, 1)
    em_preds = (probs @ dist_matrix).argmin(1)
    mp_preds = probs.argmax(1)
    results = []
    i = 0
    for y, path, em_pred, mp_pred, prob in zip(ys, paths, em_preds, mp_preds, probs):
        results.append({
            "image": path,
            "annotation_id": int(path.split('_')[-1].split('.')[0]),
            "y":class_names[y],
            "em_pred": class_names[em_pred],
            "em_conf": prob[em_pred].item(),
            "em_dist": dist_matrix[em_pred, y].item(),
            "mp_pred": class_names[mp_pred],
            "mp_conf": prob[mp_pred].item(),
            "mp_dist": dist_matrix[mp_pred, y].item(),
        })
    df = pd.DataFrame(results)
    df.sort_values('annotation_id').to_csv(save_path, index=False)
         

def run_inference(model, image_dir, class_names, device='cuda' if torch.cuda.is_available() else 'cpu'):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = InferenceDataset(image_dir, transform=transform)
    #sampler = RandomSampler(dataset)
    sampler = GroupedPrefixSampler(dataset, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=40, sampler=sampler)

    model.eval()
    model.to(device)

    results = []
    pr = []
    p = []

    with torch.no_grad():
        for batch in dataloader:
            images, paths, image_ids, img_sizes = batch['x'], batch['path'], batch['image_id'], batch['img_size']
            images = images.to(device)
            emb, outputs = model(images, image_ids, img_sizes)
            probs = torch.softmax(outputs, dim=1)

            best_choice = (probs @ model.dist_matrix).argmin(1)
            highest_choice = probs.argmax(1)

            i = 0
            for path, em_pred, mp_pred, prob in zip(paths, best_choice, highest_choice, probs):
                results.append({
                    "image": path,
                    "concept_name": class_names[em_pred],
                    "confidence": prob[em_pred].item(),
                    "MP_name": class_names[mp_pred],
                    "MP_confidence": prob[mp_pred].item(),
                    'annotation_id': int(path.split('_')[-1].split('.')[0]),
                })
                pr += [prob.cpu().detach().numpy().tolist()]
                p += [path]
            
    pr = pd.DataFrame(pr, columns=class_names, index=p)

    return results, pr


def save_image_grid(images: torch.Tensor, filename: str, nrow: int = 8):
    """
    Save a batch of images to a PNG file in a grid layout.

    Args:
        images (torch.Tensor): Tensor of shape (B, C, H, W).
        filename (str): Output filename (e.g., 'grid.png').
        nrow (int): Number of images per row in the grid.
    """
    # Make sure tensor is in the correct range
    images = images.clone().detach()
    images = (images - images.min()) / (images.max()-images.min())

    grid = vutils.make_grid(images, nrow=nrow, padding=2, normalize=True)

    # Convert to numpy and transpose for matplotlib (C, H, W) -> (H, W, C)
    np_grid = grid.cpu().numpy().transpose((1, 2, 0))

    plt.figure(figsize=(nrow, images.size(0) // nrow))
    plt.axis('off')
    plt.imshow(np_grid)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()


# modified from ete3 codebase
def my_convert_to_ultrametric(tree, tree_length=None, strategy="fixed_child"):
    """
    .. versionadded: 2.1

    Converts a tree into ultrametric topology (all leaves must have
    the same distance to root). Note that, for visual inspection
    of ultrametric trees, node.img_style["size"] should be set to
    0.
    """

    # Could something like this replace the old algorithm?
    # most_distant_leaf, tree_length = self.get_farthest_leaf()
    # for leaf in self:
    #    d = leaf.get_distance(self)
    #    leaf.dist += (tree_length - d)
    # return

    # get origin distance to root
    dist2root = {tree: 0.0}
    for node in tree.iter_descendants("levelorder"):
        dist2root[node] = dist2root[node.up] + node.dist

    # get tree length by the maximum
    if not tree_length:
        tree_length = max(dist2root.values())
    else:
        tree_length = float(tree_length)

    # converts such that starting from the leaves, each group which can have
    # the smallest step, does. This ensures all from the same genus are assumed the same
    # space apart
    if strategy == "fixed_child":
        step = 1.0

        # pre-calculate how many splits remain under each node
        node2max_depth = {}
        for node in tree.traverse("postorder"):
            if not node.is_leaf():
                max_depth = max([node2max_depth[c] for c in node.children]) + 1
                node2max_depth[node] = max_depth
            else:
                node2max_depth[node] = 1
        node2dist = {tree: -1.0}
        # modify the dist property of nodes
        for node in tree.iter_descendants("levelorder"):
            node.dist = tree_length - node2dist[node.up] - node2max_depth[node] * step

            # print(node,node.dist, node.up)
            node2dist[node] = node.dist + node2dist[node.up]

    return tree


def get_T_matrix(classes, conversion, phylogeny):
    t = phylogeny.copy()
    #t = my_convert_to_ultrametric(t)
    #for node in t.traverse():
    
    #    node.name = node.name.replace(' ','_').capitalize()
    no_classes = len(classes)
    T = np.zeros((no_classes, no_classes))

    for i in tqdm(range(len(classes))):
        leaf1 = [node for node in t.traverse() if node.name == classes[i]][0]
        j = 0
        for j in range(i, len(classes)):
            leaf2 =  [node for node in t.traverse() if node.name == classes[j]][0]
            ancestor = leaf1.get_common_ancestor(leaf2)
            T[i, j] = ancestor.get_distance(t)
            T[j, i] = T[i, j]

    #class_indices = [conversion[x] for x in classes]
    T = pd.DataFrame(T, index=classes, columns=classes)
    return T / (T.max() + 1)

def tree_to_distance_matrix(tree_file, labels):
    # Load tree
    tree = Tree(tree_file, format=3)

    n = len(labels)
    labels = sorted(labels)

    # Create a blank distance matrix
    dist_matrix = np.zeros((n, n))

    # Fill the matrix with pairwise distances
    for i, name1 in enumerate(labels):
        node1 = [node for node in tree.traverse() if node.name == name1][0]
        for j, name2 in enumerate(labels):
            if i <= j:

                d = node1.get_distance(str(name2))
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d  # symmetric

    df = pd.DataFrame(dist_matrix, index=labels, columns=labels)
    df['Zoantharia'] = df['Zoantharia'].map(lambda x: max(0, x-1))# fix for Zoantharia to match official distances

    return df

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2