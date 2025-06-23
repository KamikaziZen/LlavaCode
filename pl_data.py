import logging
import time

import pytorch_lightning as pl
from datasets import load_from_disk
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from preprocess import AST

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LLMDataset(Dataset):
    def __init__(self,
                 data,
                 ast_tokenizer=None,
                 code_tokenizer=None,
                 max_seq_length=2048,
                 pad_token_id=0,
                 # ">:<" - this is a KOSTYL, maybe add another token?
                 structure_token_id=25782):
        super(LLMDataset, self).__init__()
        self.data = data
        self.pad_token_id = pad_token_id
        self.max_seq_length = max_seq_length
        self.code_tokenizer = code_tokenizer
        self.ast_tokenizer = ast_tokenizer
        self.structure_token_id = structure_token_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind):
        # indexing the chunked data directly
        # TODO: remove this dataclass completely, make another dataset with precalculated ast
        source_tokens = torch.tensor(self.data[ind]['token_ids'])
        fim_prefix, fim_suffix, fim_middle = torch.tensor([1]), torch.tensor([3]), torch.tensor([2])
        left_context = torch.tensor(self.data[ind]['lc_token_ids'])
        right_context = torch.tensor(self.data[ind]['rc_token_ids'])
        source_tokens = torch.cat([
            fim_prefix,
            left_context,
            fim_suffix,
            right_context,
            fim_middle,
            torch.tensor([self.structure_token_id])])
        seq_length = len(source_tokens)
        # TODO: remove hardcoded seq length and pad_token_id, use data collator with padding instead?
        source_tokens = F.pad(
            source_tokens,
            (0, self.max_seq_length-seq_length),
            value=self.pad_token_id).to(torch.long)

        item = {"input_ids": source_tokens}

        if self.ast_tokenizer:
            lc_tokens = self.code_tokenizer.decode(left_context)
            ast_tokens = AST(lc_tokens, 'python', self.ast_tokenizer)
            # print('AST', ast_tokens)
            max_length = 512
            ast_tokens = ast_tokens[:max_length-4]
            ast_tokens = [self.ast_tokenizer.cls_token, "<encoder-only>", self.ast_tokenizer.sep_token] + ast_tokens + [self.ast_tokenizer.sep_token]
            ast_ids = self.ast_tokenizer.convert_tokens_to_ids(ast_tokens)
            ast_ids = torch.tensor(ast_ids)
            seq_length = len(ast_ids)
            ast_ids = F.pad(
                ast_ids,
                (0, 512-seq_length),
                value=self.ast_tokenizer.pad_token_id).to(torch.long)
            item.update(ast_ids=ast_ids)

        return item


class DataModule(pl.LightningDataModule):
    def __init__(self, data_prefix, train_datadir, valid_datadir, train_batch_size,
                 valid_batch_size, num_workers=0, code_tokenizer=None, ast_tokenizer=None):
        super(DataModule, self).__init__()
        self.data_prefix = data_prefix
        self.train_datadir = train_datadir
        self.valid_datadir = valid_datadir
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.num_workers = num_workers
        self.code_tokenizer = code_tokenizer
        self.ast_tokenizer = ast_tokenizer

        logger.info(f"Initializing DataModule w/ train_bs={self.train_batch_size}, "
                    f"valid_bs={self.valid_batch_size}")

    def setup(self, stage=None):
        '''Called by every process'''
        logger.info('Loading data...')

        train_orig_data = load_from_disk(self.train_datadir)
        self.train_data = LLMDataset(train_orig_data,
                                     code_tokenizer=self.code_tokenizer,
                                     ast_tokenizer=self.ast_tokenizer)

        valid_orig_data = load_from_disk(self.valid_datadir)
        self.valid_data = LLMDataset(valid_orig_data,
                                     code_tokenizer=self.code_tokenizer,
                                     ast_tokenizer=self.ast_tokenizer)

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
