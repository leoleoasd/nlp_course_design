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

class LitAutoEncoder(pl.LightningModule):

    def __init__(self, args = None):
        super().__init__()
        self.lr = args.lr if args is not None else 1e-5
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.mlp = nn.Sequential(
            nn.Linear(768, 50),
            nn.Tanh(),
            nn.Linear(50, 2),
        )

    def forward(self, batch):
        x, y, _ = batch
        input_ids, attention_mask, token_type_ids = x
        embedding = self.bert(input_ids, attention_mask, token_type_ids).last_hidden_state
        all_score = F.log_softmax(self.mlp(embedding), dim=-1)
        score = all_score[:, :y.size(1), :]
        score = score.reshape(-1, 2)
        attention_mask = attention_mask.reshape(-1)
        y = y.reshape(-1)
        if y.size(0) != score.size(0):
            embed()
        loss: torch.Tensor = F.nll_loss(score, y, reduction='none')
        loss.masked_fill_(attention_mask == 0, 0)
        loss = loss.mean()
        out = torch.logical_and(torch.max(score, dim=-1)[1], attention_mask)
        corr = torch.sum(out.logical_and(y))
        prec = corr / torch.sum(out)
        recall = corr / torch.sum(y)
        f1 = 2 * prec * recall / (prec + recall)
        return loss, prec, recall, f1, torch.max(all_score, dim=-1)[1]

    def training_step(self, batch, batch_idx):
        loss, prec, recall, f1, _ = self.forward(batch)
        self.log("train/prec", prec)
        self.log("train/recall", recall)
        self.log("train/f1", f1)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, prec, recall, f1, _ = self.forward(batch)
        self.log("valid/prec", prec)
        self.log("valid/recall", recall)
        self.log("valid/f1", f1)
        return loss

    def test_step(self, batch, batch_idx):
        loss, prec, recall, f1, _ = self.forward(batch)
        self.log("test/prec", prec)
        self.log("test/recall", recall)
        self.log("test/f1", f1)
        return loss

    def predict_step(self, batch, batch_idx, _ = 0) -> Any:
        loss, prec, recall, f1, out = self.forward(batch)
        return out, batch[2]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--dataset_name', type=str, default='msr')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--not_train', dest='train', default=True, action='store_false')
    parser.add_argument('--predict', dest='predict', default=False, action='store_true')
    parser.add_argument('--not_test', dest='test', default=True, action='store_false')
    parser.add_argument('--test_ckpt_path', type=str, default=None)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    data = CWSModule(args)

    # ------------
    # model
    # ------------

    # ------------
    # training
    # ------------
    logger = WandbLogger(project="nlp_course_exp", log_model='all')
    checkpoint_callback = ModelCheckpoint(save_top_k=2,
                                          monitor="valid/f1",
                                          mode='max',
                                          filename="cws-e{epoch:02d}-{step}-f1{valid/f1:.4f}",
                                          save_on_train_epoch_end=True,
                                          auto_insert_metric_name=False,
                                          )
    trainer: pl.Trainer = pl.Trainer.from_argparse_args(args, callbacks=[
        checkpoint_callback,
        EarlyStopping(monitor="valid/f1", mode="max")
    ], logger=logger, log_every_n_steps=1)

    if args.train:
        model = LitAutoEncoder(args)
        trainer.fit(model, data)
    else:
        model = LitAutoEncoder.load_from_checkpoint(args.test_ckpt_path)
    if args.test:
        result = trainer.test(model=model, datamodule=data)
        print(result)
    if args.predict:
        result = trainer.predict(model=model, datamodule=data)
        # print(result)
        embed()
        outs = []
        for res in result:
            scores = res[0][:, 1:-1]
            text = res[1]
            for s, t in zip(scores, text):
                out_text = ""
                for i, c in zip(s, t):
                    out_text += c
                    if i:
                        out_text += "  "
                if out_text.endswith("。  "):
                    out_text = out_text[:-3]
                outs.append(out_text)
        with open("out.txt", "wb") as f:
            for o in outs:
                f.write(o.encode("utf-8") + b"\r\n")

        embed()


if __name__ == '__main__':
    cli_main()
