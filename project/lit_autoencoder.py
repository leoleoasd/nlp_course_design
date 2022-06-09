from argparse import ArgumentParser
import torch
from torch import nn
import torch.nn.functional as F
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from CWSData import *
from IPython import embed
from transformers import BertModel

class LitAutoEncoder(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.mlp = nn.Linear(768, 2)

    def forward(self, x):
        input_ids, attention_mask, token_type_ids = x
        out = self.bert(input_ids, attention_mask, token_type_ids).last_hidden_state
        return F.logsigmoid(self.mlp(out))

    def training_step(self, batch, batch_idx):
        x, y = batch
        # [32, 512, 2]
        y1 = self.forward(x)
        y1 = y1[:, :y.size(1), :]
        # y: [32, xxx]
        loss = F.nll_loss(y1, y)
        out = torch.max(y1, dim=-1)[1]
        corr = torch.sum(out.logical_and(y))
        prec = corr / torch.sum(out)
        recall = corr / torch.sum(y)
        f1 = 2 * prec * recall /(prec + recall)
        self.log("train/prec", prec)
        self.log("train/recall", recall)
        self.log("train/f1", f1)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # [32, 512, 2]
        y1 = self.forward(x)
        y1 = y1[:, :y.size(1), :]
        # y: [32, xxx]
        loss = F.nll_loss(y1, y)
        out = torch.max(y1, dim=-1)[1]
        corr = torch.sum(out.logical_and(y))
        prec = corr / torch.sum(out)
        recall = corr / torch.sum(y)
        f1 = 2 * prec * recall /(prec + recall)
        self.log("valid/prec", prec)
        self.log("valid/recall", recall)
        self.log("valid/f1", f1)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        # [32, 512, 2]
        y1 = self.forward(x)
        y1 = y1[:, :y.size(1), :]
        # y: [32, xxx]
        # flat
        y1 = y1.reshape(-1, 2)
        y = y.reshape(-1)
        embed()
        loss = F.nll_loss(y1, y)
        out = torch.max(y1, dim=-1)[1]
        corr = torch.sum(out.logical_and(y))
        prec = corr / torch.sum(out)
        recall = corr / torch.sum(y)
        f1 = 2 * prec * recall /(prec + recall)
        self.log("test/prec", prec)
        self.log("test/recall", recall)
        self.log("test/f1", f1)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--dataset_name', type=str, default='msr')
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    data = CWSModule(args)

    # ------------
    # model
    # ------------
    model = LitAutoEncoder()

    # ------------
    # training
    # ------------
    logger = WandbLogger(project="nlp_course_exp")
    trainer: pl.Trainer = pl.Trainer.from_argparse_args(args, logger=logger)
    trainer.fit(model, data)

    # ------------
    # testing
    # ------------
    result = trainer.test(datamodule=data)
    print(result)


if __name__ == '__main__':
    cli_main()
