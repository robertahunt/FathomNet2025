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
from utils import run_inference, save_predictions
from model import EfficientNetClassifier

SEED = 0
CHECKPOINT = '' #'/path/to/checkpoint/epoch=95-step=67392.ckpt'
CONTINUE = False #True
name = "CutMix+Mixup+LPLLoss+001Triplet+Graph+moddedaug+modprobs"


torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

logger = TensorBoardLogger("tb_logs", name=name)
checkpoint_callback = ModelCheckpoint(dirpath=logger.log_dir, save_top_k=1, monitor="val_loss")
checkpoint_EM_callback = ModelCheckpoint(dirpath=logger.log_dir, save_top_k=1, monitor="val_EM_tree_loss")
checkpoint_latest = ModelCheckpoint(dirpath=logger.log_dir, save_top_k=1, mode='max', monitor="epoch")

if __name__ == "__main__":
    tree = Tree('data/tree.nh', format=3)
    data_module = ImageFolderDataModule("data/dataset", batch_size=40)
    class_names = data_module.train_set.classes

    if CONTINUE & (len(CHECKPOINT) > 0):
        data_module.setup_samplers()
        assert os.path.exists(CHECKPOINT)
        ckpt_name = CHECKPOINT.split('/')[-1].split('.')[0]
        model = EfficientNetClassifier.load_from_checkpoint(CHECKPOINT)
        trainer = Trainer(max_epochs=100, accelerator="auto", logger=logger, callbacks=[checkpoint_callback, checkpoint_EM_callback, checkpoint_latest])
        trainer.fit(model, datamodule=data_module)
        #model.load_from_checkpoint(CHECKPOINT)
    elif len(CHECKPOINT):
        assert os.path.exists(CHECKPOINT)
        ckpt_name = CHECKPOINT.split('/')[-1].split('.')[0]
        model = EfficientNetClassifier.load_from_checkpoint(CHECKPOINT)
        #model.load_from_checkpoint(CHECKPOINT)
        
        predictions, probs = run_inference(model, image_dir="data_test/rois", class_names=class_names)
        df = pd.DataFrame(predictions)
        df[['annotation_id','concept_name']].sort_values('annotation_id').to_csv(os.path.join(os.path.dirname(CHECKPOINT), f'{ckpt_name}_submission.csv'), index=False)
        probs.sort_index().to_csv(os.path.join(os.path.dirname(CHECKPOINT), f'{ckpt_name}_probabilities.csv'), index=True)

    else:
        data_module.setup_samplers()
        model = EfficientNetClassifier(
                        tree_path="data/tree.nh",
                        labels = class_names,
                        num_classes=79, lr=3e-4)
        trainer = Trainer(max_epochs=100, accelerator="auto", logger=logger, callbacks=[checkpoint_callback, checkpoint_EM_callback],
                          log_every_n_steps=50)
        trainer.fit(model, datamodule=data_module)



