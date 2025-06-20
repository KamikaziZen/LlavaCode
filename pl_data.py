import logging
import time

import pytorch_lightning as pl
from datasets import load_from_disk
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LLMDataset(Dataset):
    def __init__(self, data, max_seq_length=2048, pad_token_id=0):
        super(LLMDataset, self).__init__()
        self.data = data
        self.pad_token_id = pad_token_id
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind):
        # indexing the chunked data directly
        source_tokens = torch.tensor(self.data[ind]['token_ids'])
        fim_prefix = torch.tensor([1])
        fim_suffix = torch.tensor([3])
        fim_middle = torch.tensor([2])
        source_tokens = torch.cat([
            fim_prefix, 
            torch.tensor(self.data[ind]['lc_token_ids']), 
            fim_suffix, 
            torch.tensor(self.data[ind]['rc_token_ids']),
            fim_middle])
        seq_length = len(source_tokens)
        # TODO: remove hardcoded seq length and pad_token_id, use data collator with padding instead
        source_tokens = F.pad(source_tokens, (0, self.max_seq_length-seq_length), value=self.pad_token_id).to(torch.long)
        return {"input_ids": source_tokens}


class DataModule(pl.LightningDataModule):
    def __init__(self, data_prefix, train_datadir, valid_datadir, train_batch_size, 
                 valid_batch_size, num_workers=0):
        super(DataModule, self).__init__()
        self.data_prefix = data_prefix
        self.train_datadir = train_datadir
        self.valid_datadir = valid_datadir
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.num_workers = num_workers
        logger.info(f"Initializing DataModule w/ train_bs={self.train_batch_size}, "
                    f"valid_bs={self.valid_batch_size}")

    def setup(self, stage=None):
        '''Called by every process'''
        logger.info('Loading data...')


        train_orig_data = load_from_disk(self.train_datadir)
        self.train_data = LLMDataset(train_orig_data)


        valid_orig_data = load_from_disk(self.valid_datadir)
        self.valid_data = LLMDataset(valid_orig_data)

        logger.info(f'Loaded Train data with {len(self.train_data)} examples')
        logger.info(f"train_bs={self.train_batch_size}\t "
                    f"valid_bs={self.valid_batch_size}")
        time.sleep(5)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.train_batch_size, 
                          num_workers=8, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_data, batch_size=self.valid_batch_size, 
                          num_workers=8, shuffle=False)

# References
# ----------
# https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html
