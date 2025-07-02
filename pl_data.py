import logging
import time

import pytorch_lightning as pl
from datasets import load_from_disk
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import math

from preprocess import AST

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ASTLcontextDataset(Dataset):
    def __init__(self,
                 data,
                 ast_tokenizer,
                 code_tokenizer,
                 structure_token_id,
                 max_seq_length=2048,
                 max_ast_seqlen=512):
        super(ASTLcontextDataset, self).__init__()
        self.data = data
        self.max_seq_length = max_seq_length
        self.code_tokenizer = code_tokenizer
        self.ast_tokenizer = ast_tokenizer
        self.structure_token_id = self.code_tokenizer.vocab_size
        self.max_ast_seqlen = max_ast_seqlen

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind):
        # indexing the chunked data directly
        # source_tokens = torch.tensor(self.data[ind]['token_ids'])
        fim_prefix, fim_suffix, fim_middle = torch.tensor([1]), torch.tensor([3]), torch.tensor([2])
        left_context_ids = torch.tensor(self.data[ind]['lc_token_ids'])
        right_context_ids = torch.tensor(self.data[ind]['rc_token_ids'])
        target_ids = torch.tensor(self.data[ind]['tgt_token_ids'])

        left_context = self.code_tokenizer.decode(left_context_ids)
        # AST function ignores comments
        ast_tokens = AST(left_context.replace('#', ''), 'python', self.ast_tokenizer)
        patch_length = self.max_ast_seqlen - 4  # 4 special tokens for unixcoder
        num_structure_tokens = math.ceil(len(ast_tokens) / patch_length)

        structure_ids = []
        for i in range(num_structure_tokens):
            patch = ast_tokens[i * patch_length: (i + 1) * patch_length]
            patch_tokens = [self.ast_tokenizer.cls_token, "<encoder-only>", self.ast_tokenizer.sep_token] + patch + [self.ast_tokenizer.sep_token]
            patch_ids = self.ast_tokenizer.convert_tokens_to_ids(patch_tokens)
            structure_ids.extend(patch_ids)
        structure_ids = torch.tensor(structure_ids, dtype=torch.long)

        input_ids = torch.cat([
            fim_prefix,
            left_context_ids,
            torch.tensor([self.structure_token_id] * num_structure_tokens),
            fim_suffix,
            right_context_ids,
            fim_middle,
            target_ids]).to(torch.long)

        item = {"input_ids": input_ids, 'structure_ids': structure_ids, 'num_structure_tokens': num_structure_tokens}

        return item


class CodeCfcDataset(Dataset):
    def __init__(self,
                 data,
                 ast_tokenizer,
                 code_tokenizer,
                 structure_token_id,
                 max_seq_length=2048,
                 max_ast_seqlen=512):
        super(CodeCfcDataset, self).__init__()
        self.data = data
        self.max_seq_length = max_seq_length
        self.code_tokenizer = code_tokenizer
        self.ast_tokenizer = ast_tokenizer
        self.structure_token_id = self.code_tokenizer.vocab_size
        self.max_ast_seqlen = max_ast_seqlen

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind):
        # indexing the chunked data directly
        # source_tokens = torch.tensor(self.data[ind]['token_ids'])
        fim_prefix, fim_suffix, fim_middle = torch.tensor([1]), torch.tensor([3]), torch.tensor([2])
        left_context_ids = torch.tensor(self.data[ind]['lc_token_ids'])
        right_context_ids = torch.tensor(self.data[ind]['rc_token_ids'])
        target_ids = torch.tensor(self.data[ind]['tgt_token_ids'])

        structure_tokens = self.ast_tokenizer.tokenize(
            self.code_tokenizer.decode(self.data[ind]['cfc_token_ids']))
        patch_length = self.max_ast_seqlen - 4  # 4 special tokens for unixcoder
        num_structure_tokens = math.ceil(len(structure_tokens) / patch_length)

        structure_ids = []
        for i in range(num_structure_tokens):
            patch = structure_tokens[i * patch_length: (i + 1) * patch_length]
            patch_tokens = [self.ast_tokenizer.cls_token, "<encoder-only>", self.ast_tokenizer.sep_token] \
                + patch + [self.ast_tokenizer.sep_token]
            patch_ids = self.ast_tokenizer.convert_tokens_to_ids(patch_tokens)
            structure_ids.extend(patch_ids)
        structure_ids = torch.tensor(structure_ids, dtype=torch.long)

        input_ids = torch.cat([
            torch.tensor([self.structure_token_id] * num_structure_tokens),
            fim_prefix,
            left_context_ids,
            fim_suffix,
            right_context_ids,
            fim_middle,
            target_ids]).to(torch.long)

        item = {"input_ids": input_ids, 'structure_ids': structure_ids, 'num_structure_tokens': num_structure_tokens}
        return item


class ASTCfcDataset(Dataset):
    def __init__(self,
                 data,
                 ast_tokenizer,
                 code_tokenizer,
                 structure_token_id,
                 max_seq_length=2048,
                 max_ast_seqlen=512):
        super(ASTCfcDataset, self).__init__()
        self.data = data
        self.max_seq_length = max_seq_length
        self.code_tokenizer = code_tokenizer
        self.ast_tokenizer = ast_tokenizer
        self.structure_token_id = self.code_tokenizer.vocab_size
        self.max_ast_seqlen = max_ast_seqlen

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind):
        # indexing the chunked data directly
        # source_tokens = torch.tensor(self.data[ind]['token_ids'])
        fim_prefix, fim_suffix, fim_middle = torch.tensor([1]), torch.tensor([3]), torch.tensor([2])
        left_context_ids = torch.tensor(self.data[ind]['lc_token_ids'])
        right_context_ids = torch.tensor(self.data[ind]['rc_token_ids'])
        cfc_ids = torch.tensor(self.data[ind]['cfc_token_ids'])
        target_ids = torch.tensor(self.data[ind]['tgt_token_ids'])

        cfc = self.code_tokenizer.decode(cfc_ids)
        # AST function ignores comments
        ast_tokens = AST(cfc.replace('#', ''), 'python', self.ast_tokenizer)
        patch_length = self.max_ast_seqlen - 4  # 4 special tokens for unixcoder
        num_structure_tokens = math.ceil(len(ast_tokens) / patch_length)

        structure_ids = []
        for i in range(num_structure_tokens):
            patch = ast_tokens[i * patch_length: (i + 1) * patch_length]
            patch_tokens = [self.ast_tokenizer.cls_token, "<encoder-only>", self.ast_tokenizer.sep_token] \
                + patch + [self.ast_tokenizer.sep_token]
            patch_ids = self.ast_tokenizer.convert_tokens_to_ids(patch_tokens)
            structure_ids.extend(patch_ids)
        structure_ids = torch.tensor(structure_ids, dtype=torch.long)

        input_ids = torch.cat([
            torch.tensor([self.structure_token_id] * num_structure_tokens),
            fim_prefix,
            left_context_ids,
            fim_suffix,
            right_context_ids,
            fim_middle,
            target_ids]).to(torch.long)

        item = {"input_ids": input_ids, 'structure_ids': structure_ids, 'num_structure_tokens': num_structure_tokens}
        return item


class LlavaCodeDataCollator:
    def __init__(self, code_tokenizer, ast_tokenizer):
        self.code_tokenizer = code_tokenizer
        self.ast_tokenizer = ast_tokenizer
        # remove these checks later
        assert self.ast_tokenizer.pad_token_id == 1
        assert self.code_tokenizer.pad_token_id == 0

    def __call__(self, features):
        input_ids = [{'input_ids': f['input_ids']} for f in features]
        structure_ids = [{'input_ids': f['structure_ids']} for f in features]

        batch = self.code_tokenizer.pad(
            input_ids,
            padding=True,
            return_tensors='pt',
            padding_side='right'
        )

        ast_batch = self.ast_tokenizer.pad(
            structure_ids,
            padding=True,
            pad_to_multiple_of=512,
            return_tensors='pt',
            padding_side='right'
        )

        batch['structure_ids'] = ast_batch['input_ids']
        batch['num_structure_tokens'] = torch.tensor([f['num_structure_tokens'] for f in features], dtype=torch.int)

        return batch


class DataModule(pl.LightningDataModule):
    def __init__(self, data_prefix, train_datadir, valid_datadir, train_batch_size,
                 valid_batch_size, structure_token_id, code_tokenizer,
                 ast_tokenizer, num_workers=0,):
        super(DataModule, self).__init__()
        self.data_prefix = data_prefix
        self.train_datadir = train_datadir
        self.valid_datadir = valid_datadir
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.num_workers = num_workers
        self.code_tokenizer = code_tokenizer
        self.ast_tokenizer = ast_tokenizer
        self.structure_token_id = structure_token_id

        self.data_collator = LlavaCodeDataCollator(code_tokenizer, ast_tokenizer)

        logger.info(f"Initializing DataModule w/ train_bs={self.train_batch_size}, "
                    f"valid_bs={self.valid_batch_size}")

    def get_dataset(self, raw_data):
        if self.data_prefix == 'ast_lcontext':
            return ASTLcontextDataset(
                raw_data,
                code_tokenizer=self.code_tokenizer,
                ast_tokenizer=self.ast_tokenizer,
                structure_token_id=self.structure_token_id)
        elif self.data_prefix == 'ast_cfc':
            return ASTCfcDataset(
                raw_data,
                code_tokenizer=self.code_tokenizer,
                ast_tokenizer=self.ast_tokenizer,
                structure_token_id=self.structure_token_id)
        elif self.data_prefix == 'code_cfc':
            return CodeCfcDataset(
                raw_data,
                code_tokenizer=self.code_tokenizer,
                ast_tokenizer=self.ast_tokenizer,
                structure_token_id=self.structure_token_id)
        else:
            raise ValueError(f'Invalid data_prefix: {self.data_prefix}')

    def setup(self, stage=None):
        '''Called by every process'''
        logger.info('Loading data...')

        train_orig_data = load_from_disk(self.train_datadir)
        valid_orig_data = load_from_disk(self.valid_datadir)

        self.train_data = self.get_dataset(train_orig_data)
        self.valid_data = self.get_dataset(valid_orig_data)

        logger.info(f'Loaded Train data with {len(self.train_data)} examples')
        logger.info(f"train_bs={self.train_batch_size}\t "
                    f"valid_bs={self.valid_batch_size}")
        time.sleep(5)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.train_batch_size,
                          collate_fn=self.data_collator, num_workers=8, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_data, batch_size=self.valid_batch_size,
                          collate_fn=self.data_collator, num_workers=8, shuffle=False)

# References
# ----------
# https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html
