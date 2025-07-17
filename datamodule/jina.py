import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import random

from .utils import pack_fim_inputs


class JinaDataset(Dataset):
    """Dataset type: n chunks of cross-file context, m lines each, stored as an array
    """
    def __init__(self,
                 data,
                 training_stage,
                 structure_tokenizer,
                 code_tokenizer,
                 structure_token_id,
                 fim_tokens_ids,
                 num_structure_tokens,
                 max_seq_length=2048,
                 max_structure_length=512,
                 lc_rc_ratio=2.0):
        super(JinaDataset, self).__init__()
        self.data = data
        self.max_seq_length = max_seq_length
        self.code_tokenizer = code_tokenizer
        self.structure_tokenizer = structure_tokenizer
        self.num_structure_tokens = num_structure_tokens
        self.structure_token_id = structure_token_id
        self.max_structure_length = max_structure_length
        self.lc_rc_ratio = lc_rc_ratio
        self.fim_tokens_ids = fim_tokens_ids
        self.training_stage = training_stage

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind):

        if self.training_stage == 0:

            # instead of figuring out how to unravel this into 10x bigger dataset, just choose a random and increase number of epochs
            chunk = random.choice(self.data[ind]['content']['crossfile_array'])
            cfc = '\n'.join(chunk.splitlines()[1:])  # removing file path in the first line

            input_ids = self.code_tokenizer(cfc).input_ids  # turning cfc to tokens of the language model

            cfc_tokens = self.structure_tokenizer.tokenize(cfc)  # TODO: try decommenting?
            cfc_tokens = cfc_tokens[:self.max_structure_length]
            chunk_ids = self.structure_tokenizer.convert_tokens_to_ids(cfc_tokens)

            item = {"input_ids": input_ids, 'structure_ids': chunk_ids}

        else:

            left_context_ids = self.code_tokenizer(self.data[ind]['content']['prompt'], return_tensors='pt').input_ids[0]
            right_context_ids = self.code_tokenizer(self.data[ind]['content']['right_context'], return_tensors='pt').input_ids[0]
            target_ids = self.code_tokenizer(self.data[ind]['content']['groundtruth'], return_tensors='pt').input_ids[0]

            tgt_len = len(target_ids)
            lr_budget = self.max_seq_length - tgt_len - self.num_structure_tokens - 3  # 3 tokens for FIM
            rc_budget = int(lr_budget / (self.lc_rc_ratio + 1))
            lc_budget = int(rc_budget * self.lc_rc_ratio)

            left_context_ids = left_context_ids[-lc_budget:]
            right_context_ids = right_context_ids[:rc_budget]

            structure_ids = []
            for chunk in self.data[ind]['content']['crossfile_array'][:self.num_structure_tokens]:
                cfc = '\n'.join(chunk.splitlines()[1:])  # removing file path in the first line
                cfc_tokens = self.structure_tokenizer.tokenize(cfc)  # TODO: try decommenting?
                cfc_tokens = cfc_tokens[:self.max_structure_length]
                chunk_ids = self.structure_tokenizer.convert_tokens_to_ids(cfc_tokens)
                structure_ids.extend(F.pad(torch.tensor(chunk_ids), (0, self.max_structure_length-len(chunk_ids)), value=self.structure_tokenizer.pad_token_id))
            structure_ids = torch.tensor(structure_ids, dtype=torch.long)

            input_ids = pack_fim_inputs(
                    self.fim_tokens_ids, left_context_ids, right_context_ids, target_ids, self.structure_token_id, self.num_structure_tokens)

            item = {"input_ids": input_ids, 'structure_ids': structure_ids, 'num_structure_tokens': self.num_structure_tokens}

        return item
