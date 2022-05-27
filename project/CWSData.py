import torch
from transformers import BertTokenizer
import pytorch_lightning as pl
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader
from IPython import embed


class CWSDataset(torch.utils.data.IterableDataset):
    def __init__(self, file):
        super().__init__()
        self.file = file

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            with open(self.file) as f:
                d = f.readlines()
            yield from d
        else:
            count = 0
            with open(self.file) as f:
                d = f.readlines()
            for i in d:
                if count % worker_info.num_workers == worker_info.id:
                    yield i
                count += 1


class CWSModule(pl.LightningDataModule):

    def __init__(self, args):
        super().__init__()
        self.val_dataset = None
        self.train_dataset = None
        self.test_dataset = None
        self.predict_dataset = None
        # self.vocab = None
        self.tokenizer = None
        # self.vocab_size = 0
        self.batch_size = args.batch_size
        # self.vocab_dir = args.vocab_dir
        self.dataset_name = args.dataset_name

    def setup(self, stage=None):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.train_dataset = CWSDataset(
            f"data/training/{self.dataset_name}_training.utf8_train")
        self.val_dataset = CWSDataset(
            f"data/training/{self.dataset_name}_training.utf8_valid")
        self.test_dataset = CWSDataset(
            f"data/gold/{self.dataset_name}_test_gold.utf8")
        self.predict_dataset = CWSDataset(
            f"data/testing/{self.dataset_name}_test.utf8")

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            num_workers=1,
            collate_fn=self.collate(),
            batch_size=self.batch_size
        )

    def collate(self):
        def _collate(batch: list[str]):
            segments = []
            input_ids = []
            attention_mask = []
            token_type_ids = []
            for s in batch:
                s = s.rstrip().split("  ")
                # Prepend a zero for [CLS] token
                seg = [0]
                input_id = [self.tokenizer.convert_tokens_to_ids('[CLS]')]
                for i in s:
                    for j in i:
                        input_id.append(
                            self.tokenizer.convert_tokens_to_ids(j)
                        )
                        seg.append(0)
                    seg[-1] = 1
                input_id.append(self.tokenizer.convert_tokens_to_ids('[SEP]'))
                input_id = torch.LongTensor(input_id)
                input_ids.append(input_id)
                attention_mask.append(torch.ones_like(input_id))
                token_type_ids.append(torch.zeros_like(input_id))

                seg.append(0)
                segments.append(torch.LongTensor(seg))
            segments = pad_sequence(segments, True)
            input_ids = pad_sequence(input_ids, True, self.tokenizer.convert_tokens_to_ids('[PAD]'))
            attention_mask = pad_sequence(attention_mask, True)
            token_type_ids = pad_sequence(token_type_ids, True)
            segments = segments[:, :512]
            input_ids = input_ids[:, :512]
            attention_mask = attention_mask[:, :512]
            token_type_ids = token_type_ids[:, :512]
            return ((
                        input_ids,
                        attention_mask,
                        token_type_ids,
                    ),
                    segments,
                    [i.rstrip() for i in batch]
                    )

        return _collate

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            num_workers=1,
            collate_fn=self.collate(),
            batch_size=self.batch_size
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            num_workers=1,
            collate_fn=self.collate(),
            batch_size=self.batch_size
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            num_workers=1,
            collate_fn=self.collate(),
            batch_size=self.batch_size
        )
