from argparse import ArgumentParser
from typing import Any

import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch import nn
import torch.nn.functional as F
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from CWSData import *
from IPython import embed
from transformers import BertModel

bert = BertModel.from_pretrained('bert-base-chinese')

dataset = CWSDataset(
            f"data/training/msr_training.utf8")

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

outs = []
bert = bert.cuda()

for data in dataset:
    print(data)