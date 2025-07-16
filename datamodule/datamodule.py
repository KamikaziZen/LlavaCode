import logging
import time

from datasets import load_from_disk
from lightning.pytorch import LightningDataModule
import torch
from torch.utils.data import DataLoader

from .unixcoder import (
    AstLcontextDataset,
    AstCfcDataset,
    CodeCfcDataset,
    CodeAstCfcDataset,
)
from .graphcodebert import DfgDataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LlavaCodeDataCollator:
    def __init__(self, code_tokenizer, structure_tokenizer):
        self.code_tokenizer = code_tokenizer
        self.structure_tokenizer = structure_tokenizer

    def __call__(self, features):
        input_ids = [{'input_ids': f['input_ids']} for f in features]
        structure_ids = [{'input_ids': f['structure_ids']} for f in features]

        batch = self.code_tokenizer.pad(
            input_ids,
            padding=True,
            return_tensors='pt',
            padding_side='right'
        )

        # no need in case of fixed num_structure_tokens, but left for compatibility
        structure_batch = self.structure_tokenizer.pad(
            structure_ids,
            padding=True,
            pad_to_multiple_of=512,
            return_tensors='pt',
            padding_side='right'
        )
        batch['structure_ids'] = structure_batch['input_ids']

        if 'num_structure_tokens' in features[0].keys():
            batch['num_structure_tokens'] = torch.tensor([f['num_structure_tokens'] for f in features], dtype=torch.int)

        if 'structure_pos_idx' in features[0].keys():
            batch['structure_pos_idx'] = torch.vstack([f['structure_pos_idx'] for f in features])  # assumed to be  already padded
        if 'structure_attn_mask' in features[0].keys():
            batch['structure_attn_mask'] = torch.vstack([f['structure_attn_mask'] for f in features])  # assumed to be  already padded

        return batch


class LlavaCodeDataModule(LightningDataModule):
    def __init__(self, data_prefix, train_datadir, valid_datadir, train_batch_size,
                 valid_batch_size, code_tokenizer, structure_tokenizer, structure_token_id,
                 fim_tokens, training_stage, num_structure_tokens=5, num_workers=0,):
        super(LlavaCodeDataModule, self).__init__()
        self.data_prefix = data_prefix
        self.train_datadir = train_datadir
        self.valid_datadir = valid_datadir
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.num_workers = num_workers
        self.code_tokenizer = code_tokenizer
        self.structure_tokenizer = structure_tokenizer
        self.structure_token_id = structure_token_id
        self.num_structure_tokens = num_structure_tokens

        self.fim_tokens = fim_tokens
        self.fim_tokens_ids = torch.tensor(self.code_tokenizer.convert_tokens_to_ids(fim_tokens))

        print('FIM tokens:', fim_tokens)

        self.data_collator = LlavaCodeDataCollator(code_tokenizer, structure_tokenizer)

        self.training_stage = training_stage

        logger.info(f"Initializing DataModule w/ train_bs={self.train_batch_size}, "
                    f"valid_bs={self.valid_batch_size}")

    def get_dataset(self, raw_data, training_stage):
        if self.data_prefix == 'ast_lcontext':
            return AstLcontextDataset(
                raw_data,
                code_tokenizer=self.code_tokenizer,
                ast_tokenizer=self.structure_tokenizer,
                structure_token_id=self.structure_token_id,
                max_structure_length=512)
        elif self.data_prefix == 'ast_cfc':
            return AstCfcDataset(
                raw_data,
                training_stage=training_stage,
                code_tokenizer=self.code_tokenizer,
                ast_tokenizer=self.structure_tokenizer,
                fim_tokens_ids=self.fim_tokens_ids,
                structure_token_id=self.structure_token_id,
                num_structure_tokens=self.num_structure_tokens,
                max_structure_length=512)
        elif self.data_prefix == 'code_cfc':
            return CodeCfcDataset(
                raw_data,
                code_tokenizer=self.code_tokenizer,
                ast_tokenizer=self.structure_tokenizer,
                structure_token_id=self.structure_token_id,
                max_structure_length=512)
        elif self.data_prefix == 'codeast_cfc':
            return CodeAstCfcDataset(
                raw_data,
                code_tokenizer=self.code_tokenizer,
                ast_tokenizer=self.structure_tokenizer,
                structure_token_id=self.structure_token_id,
                max_structure_length=512)
        elif self.data_prefix == 'dfg_cfc':
            return DfgDataset(
                raw_data,
                code_tokenizer=self.code_tokenizer,
                dfg_tokenizer=self.structure_tokenizer,
                structure_token_id=self.structure_token_id,
                num_structure_tokens=self.num_structure_tokens,
                max_structure_length=512)
        # elif self.data_prefix == 'graph_cfc':
        #     return GraphCfcDataset(
        #         raw_data,
        #         code_tokenizer=self.code_tokenizer,
        #         structure_token_id=self.structure_token_id,
        #         max_structure_length=512)
        else:
            raise ValueError(f'Invalid data_prefix: {self.data_prefix}')

    def setup(self, stage=None):
        '''Called by every process'''
        logger.info('Loading data...')

        train_orig_data = load_from_disk(self.train_datadir)
        valid_orig_data = load_from_disk(self.valid_datadir)

        self.train_data = self.get_dataset(train_orig_data, training_stage=self.training_stage)
        self.valid_data = self.get_dataset(valid_orig_data, training_stage=self.training_stage)

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
