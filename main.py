import os
import random
import numpy as np
import pandas as pd
from ete3 import Tree

import torch

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from dataset import ImageFolderDataModule, InferenceDataset
from utils import run_inference, save_opts
from model import EfficientNetClassifier

opt = {}

opt['name'] = "Baseline+DataAug+Graph"
opt['DataAug'] = True # Use cutmix and mixup augmentation
opt['Graph'] = True # Use Graph Neural Network Layer before Final Classifier
opt['Triplet'] = 0 # multiplier for triplet loss
opt['LPL'] = 0 # multiplier for loss per level (HXE loss)
opt['SEED'] = 0 # Random seed for reproducibility
opt['CHECKPOINT'] = '' #'/path/to/checkpoint/epoch=95-step=67392.ckpt'
opt['CONTINUE'] = False # Continue training from checkpoint?


torch.manual_seed(opt['SEED'])
torch.cuda.manual_seed(opt['SEED'])
random.seed(opt['SEED'])
np.random.seed(opt['SEED'])
pl.seed_everything(opt['SEED'], workers=True)

logger = TensorBoardLogger("tb_logs", name=opt['name'])
checkpoint_callback = ModelCheckpoint(dirpath=logger.log_dir, save_top_k=1, monitor="val_loss")
checkpoint_EM_callback = ModelCheckpoint(dirpath=logger.log_dir, save_top_k=1, monitor="val_EM_tree_loss")
checkpoint_latest = ModelCheckpoint(dirpath=logger.log_dir, save_top_k=1, mode='max', monitor="epoch")

if not os.path.exists(logger.log_dir):
    os.makedirs(logger.log_dir)
save_opts(opt, os.path.join(logger.log_dir,'opt.json'))

if __name__ == "__main__":
    tree = Tree('data/tree.nh', format=3)
    data_module = ImageFolderDataModule("data/dataset", batch_size=40, opt=opt)
    class_names = data_module.train_set.classes

    if opt['CONTINUE'] & (len(opt['CHECKPOINT']) > 0):
        data_module.setup_samplers()
        assert os.path.exists(opt['CHECKPOINT'])
        ckpt_name = opt['CHECKPOINT'].split('/')[-1].split('.')[0]
        model = EfficientNetClassifier.load_from_checkpoint(opt['CHECKPOINT'])
        trainer = Trainer(max_epochs=100, accelerator="auto", logger=logger, deterministic=True, callbacks=[checkpoint_callback, checkpoint_EM_callback], log_every_n_steps=50)
        trainer.fit(model, datamodule=data_module)
    elif len(opt['CHECKPOINT']):
        assert os.path.exists(opt['CHECKPOINT'])
        ckpt_name = opt['CHECKPOINT'].split('/')[-1].split('.')[0]
        model = EfficientNetClassifier.load_from_checkpoint(opt['CHECKPOINT'])
        
        predictions, probs = run_inference(model, image_dir="data_test/rois", class_names=class_names)
        df = pd.DataFrame(predictions)
        df[['annotation_id','concept_name']].sort_values('annotation_id').to_csv(os.path.join(os.path.dirname(opt['CHECKPOINT']), f'{ckpt_name}_submission.csv'), index=False)
        probs.sort_index().to_csv(os.path.join(os.path.dirname(opt['CHECKPOINT']), f'{ckpt_name}_probabilities.csv'), index=True)

    else:
        data_module.setup_samplers()
        model = EfficientNetClassifier(
                        tree_path="data/tree.nh",
                        labels = class_names,
                        num_classes=79, lr=3e-4, opt=opt)
        trainer = Trainer(max_epochs=100, accelerator="auto", logger=logger, deterministic=True, callbacks=[checkpoint_callback, checkpoint_EM_callback],
                          log_every_n_steps=50)
        trainer.fit(model, datamodule=data_module)



