import torch
from transformers import BertTokenizer
import pytorch_lightning as pl
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader


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
            f"data/testing/{self.dataset_name}_testing.utf8")

    def collate(self):
        def _collate(batch: list[str]):
            texts = []
            segments = []
            for s in batch:
                s = s.rstrip().split("  ")
                r = ""
                # Prepend a zero for [CLS] token
                seg = [0]
                for i in s:
                    for j in i:
                        r += j
                    seg += [0 for _ in range(len(i))]
                    seg[-1] = 1
                texts.append(r)
                segments.append(torch.LongTensor(seg))
            segments = pad_sequence(segments, True)
            segments = segments[:, :512]

            tokenized = self.tokenizer(text=texts,
                                       return_tensors="pt",
                                       max_length=512,
                                       padding=True,
                                       truncation="only_first")
            return ((
                        tokenized.input_ids,
                        tokenized.attention_mask,
                        tokenized.token_type_ids,
                    ),
                    segments,
                    # batch
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
