import logging
import time

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import math
import random
from preprocess import AST

from .utils import pack_fim_inputs
from .const import PARAPHRASE_CUES

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AstLcontextDataset(Dataset):
    def __init__(self,
                 data,
                 ast_tokenizer,
                 code_tokenizer,
                 structure_token_id,
                 max_seq_length=2048,
                 max_structure_length=512):
        super(AstLcontextDataset, self).__init__()
        self.data = data
        self.max_seq_length = max_seq_length
        self.code_tokenizer = code_tokenizer
        self.ast_tokenizer = ast_tokenizer
        self.structure_token_id = structure_token_id
        self.max_structure_length = max_structure_length

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
        patch_length = self.max_structure_length - 4  # 4 special tokens for unixcoder
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


# class CodeCfcDataset_old(Dataset):
#     """Dataset type: 3 chunks of cross-file context, 40 lines each, merged into a single string
#     """
#     def __init__(self,
#                  data,
#                  ast_tokenizer,
#                  code_tokenizer,
#                  structure_token_id,
#                  max_seq_length=2048,
#                  max_structure_length=512):
#         super(CodeCfcDataset, self).__init__()
#         self.data = data
#         self.max_seq_length = max_seq_length
#         self.code_tokenizer = code_tokenizer
#         self.ast_tokenizer = ast_tokenizer
#         self.structure_token_id = structure_token_id
#         self.max_structure_length = max_structure_length

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, ind):
#         # indexing the chunked data directly
#         # source_tokens = torch.tensor(self.data[ind]['token_ids'])
#         fim_prefix, fim_suffix, fim_middle = torch.tensor([1]), torch.tensor([3]), torch.tensor([2])
#         left_context_ids = torch.tensor(self.data[ind]['lc_token_ids'])
#         right_context_ids = torch.tensor(self.data[ind]['rc_token_ids'])
#         target_ids = torch.tensor(self.data[ind]['tgt_token_ids'])

#         structure_tokens = self.ast_tokenizer.tokenize(
#             self.code_tokenizer.decode(self.data[ind]['cfc_token_ids']))
#         patch_length = self.max_structure_length - 4  # 4 special tokens for unixcoder
#         num_structure_tokens = math.ceil(len(structure_tokens) / patch_length)

#         structure_ids = []
#         for i in range(num_structure_tokens):
#             patch = structure_tokens[i * patch_length: (i + 1) * patch_length]
#             patch_tokens = [self.ast_tokenizer.cls_token, "<encoder-only>", self.ast_tokenizer.sep_token] \
#                 + patch + [self.ast_tokenizer.sep_token]
#             patch_ids = self.ast_tokenizer.convert_tokens_to_ids(patch_tokens)
#             structure_ids.extend(patch_ids)
#         structure_ids = torch.tensor(structure_ids, dtype=torch.long)

#         input_ids = torch.cat([
#             fim_prefix,
#             left_context_ids,
#             fim_suffix,
#             right_context_ids,
#             torch.tensor([self.structure_token_id] * num_structure_tokens),
#             fim_middle,
#             target_ids]).to(torch.long)

#         item = {"input_ids": input_ids, 'structure_ids': structure_ids, 'num_structure_tokens': num_structure_tokens}
#         return item


class CodeCfcDataset(Dataset):
    """Dataset type: 10 chunks of cross-file context, 10 lines each, stored as an array
    """
    def __init__(self,
                 data,
                 training_stage,
                 ast_tokenizer,
                 code_tokenizer,
                 structure_token_id,
                 fim_tokens_ids,
                 num_structure_tokens=None,
                 max_seq_length=2048,
                 max_structure_length=512,
                 lc_rc_ratio=2.0):
        super(CodeCfcDataset, self).__init__()
        # self.data = data
        print('Dataset samples before: ', len(data))
        remove_indices = [4689, 4690, 10037, 10998, 13381, 14865, 15364, 17490, 20118, 32910, 39973, 41641, 46718, 58023, 58643, 58856, 64421, 68036, 68990, 72104, 72105, 72690, 72691, 73997, 75598, 81033, 88847, 90688, 93281, 95307, 95500, 95606, 107740, 107742, 114358, 115832, 120194, 134935, 136458]
        keep_indices = [i for i in range(len(data)) if i not in remove_indices]
        self.data = data.select(keep_indices)
        print('Dataset samples: ', len(self.data))
        self.max_seq_length = max_seq_length
        self.code_tokenizer = code_tokenizer
        self.ast_tokenizer = ast_tokenizer
        self.structure_token_id = structure_token_id
        self.max_structure_length = max_structure_length
        self.lc_rc_ratio = lc_rc_ratio
        self.num_structure_tokens = num_structure_tokens
        self.training_stage = training_stage
        self.fim_tokens_ids = fim_tokens_ids
        self.expand_factor = 1
        if self.training_stage == 0:
            self.expand_factor = len(self.data[0]['content']['crossfile_array'])

    def __len__(self):

        return len(self.data) * self.expand_factor

    def __getitem__(self, ind):

        fim_prefix_id, fim_suffix_id, fim_middle_id = self.fim_tokens_ids.reshape(3, 1)

        if self.training_stage == 0:

            orig_ind = ind // self.expand_factor
            sub_ind = ind % self.expand_factor

            cfc = self.data[orig_ind]['content']['crossfile_array'][sub_ind]
            cfc = '\n'.join(cfc.splitlines()[1:])  # removing file path in the first line
            cfc_tokens = self.ast_tokenizer.tokenize(cfc)
            cfc_tokens = cfc_tokens[:self.max_structure_length - 4]  # 4 special tokens for unixcoder
            cfc_tokens = [self.ast_tokenizer.cls_token, "<encoder-only>", self.ast_tokenizer.sep_token] \
                + cfc_tokens + [self.ast_tokenizer.sep_token]
            cfc_ids = self.ast_tokenizer.convert_tokens_to_ids(cfc_tokens)
            target_ids = self.code_tokenizer(cfc, return_tensors='pt', truncation=True, max_length=self.max_seq_length).input_ids[0]

            cue = random.choice(PARAPHRASE_CUES)
            input_ids = torch.cat([
                fim_prefix_id,
                torch.tensor([self.structure_token_id]),
                self.code_tokenizer(cue, return_tensors='pt').input_ids[0],
                fim_suffix_id,
                fim_middle_id,
                target_ids]).to(torch.long)

            item = {"input_ids": input_ids, 'structure_ids': cfc_ids}

        else:

            # preliminary truncating to save memory and avoid warnings
            left_context_ids = self.code_tokenizer(self.data[ind]['content']['prompt'], return_tensors='pt', truncation=True, max_length=self.max_seq_length).input_ids[0]
            right_context_ids = self.code_tokenizer(self.data[ind]['content']['right_context'], return_tensors='pt', truncation=True, max_length=self.max_seq_length).input_ids[0]
            target_ids = self.code_tokenizer(self.data[ind]['content']['groundtruth'], return_tensors='pt', truncation=True, max_length=self.max_seq_length).input_ids[0]

            tgt_len = len(target_ids)
            lr_budget = self.max_seq_length - tgt_len - self.num_structure_tokens - 3  # 3 tokens for FIM
            rc_budget = int(lr_budget / (self.lc_rc_ratio + 1))
            lc_budget = int(rc_budget * self.lc_rc_ratio)

            left_context_ids = left_context_ids[-lc_budget:]
            right_context_ids = right_context_ids[:rc_budget]

            structure_ids = []
            for chunk in self.data[ind]['content']['crossfile_array'][:self.num_structure_tokens]:
                cfc = '\n'.join(chunk.splitlines()[1:])  # removing file path in the first line
                # cfc_ids = self.structure_tokenizer(cfc, return_tensors='pt', truncation=True, max_length=self.max_structure_length).input_ids[0]
                code_tokens = self.ast_tokenizer.tokenize(cfc)  # TODO: try decommenting?
                code_tokens = code_tokens[:self.max_structure_length - 4]  # 4 special tokens for unixcoder
                chunk_tokens = [self.ast_tokenizer.cls_token, "<encoder-only>", self.ast_tokenizer.sep_token] \
                    + code_tokens + [self.ast_tokenizer.sep_token]
                chunk_ids = self.ast_tokenizer.convert_tokens_to_ids(chunk_tokens)
                structure_ids.extend(F.pad(torch.tensor(chunk_ids), (0, self.max_structure_length-len(chunk_ids)), value=self.ast_tokenizer.pad_token_id))
            structure_ids = torch.tensor(structure_ids, dtype=torch.long)

            input_ids = torch.cat([
                fim_prefix_id,
                left_context_ids,
                fim_suffix_id,
                right_context_ids,
                self.code_tokenizer('# Here are some relevant code fragments from other files of the repo:', return_tensors='pt').input_ids[0],
                torch.tensor([self.structure_token_id] * self.num_structure_tokens),
                fim_middle_id,
                target_ids]).to(torch.long)

            # for KL-div training
            all_cfc = '\n'.join(self.data[ind]['content']['crossfile_array'][:self.num_structure_tokens])
            all_cfc_ids = self.code_tokenizer(all_cfc, return_tensors='pt', truncation=True, max_length=self.max_seq_length).input_ids[0]
            teacher_input_ids = torch.cat([
                fim_prefix_id,
                left_context_ids,
                fim_suffix_id,
                right_context_ids,
                self.code_tokenizer('# Here are some relevant code fragments from other files of the repo:', return_tensors='pt').input_ids[0],
                all_cfc_ids,
                fim_middle_id,
                target_ids]).to(torch.long)

            item = {"input_ids": input_ids, 'teacher_input_ids': teacher_input_ids, 'structure_ids': structure_ids, 'num_structure_tokens': self.num_structure_tokens}

        return item


# class AstCfcDataset_old(Dataset):
#     """Dataset type: 3 chunks of cross-file context, 40 lines each, merged into a single string
#     """
#     def __init__(self,
#                  data,
#                  ast_tokenizer,
#                  code_tokenizer,
#                  structure_token_id,
#                  fim_tokens_ids,
#                  max_seq_length=2048,
#                  max_structure_length=512,
#                  **kwargs):
#         super(AstCfcDataset, self).__init__()
#         self.data = data
#         self.max_seq_length = max_seq_length
#         self.code_tokenizer = code_tokenizer
#         self.ast_tokenizer = ast_tokenizer
#         self.structure_token_id = structure_token_id
#         self.max_structure_length = max_structure_length
#         self.fim_tokens_ids = fim_tokens_ids

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, ind):

#         left_context_ids = torch.tensor(self.data[ind]['lc_token_ids'])
#         right_context_ids = torch.tensor(self.data[ind]['rc_token_ids'])
#         cfc_ids = torch.tensor(self.data[ind]['cfc_token_ids'])
#         target_ids = torch.tensor(self.data[ind]['tgt_token_ids'])

#         cfc = self.code_tokenizer.decode(cfc_ids)
#         # AST function ignores comments
#         ast_tokens = AST(cfc.replace('#', ''), 'python', self.ast_tokenizer)
#         patch_length = self.max_structure_length - 4  # 4 special tokens for unixcoder
#         num_structure_tokens = math.ceil(len(ast_tokens) / patch_length)

#         structure_ids = []
#         for i in range(num_structure_tokens):
#             patch = ast_tokens[i * patch_length: (i + 1) * patch_length]
#             patch_tokens = [self.ast_tokenizer.cls_token, "<encoder-only>", self.ast_tokenizer.sep_token] \
#                 + patch + [self.ast_tokenizer.sep_token]
#             patch_ids = self.ast_tokenizer.convert_tokens_to_ids(patch_tokens)
#             structure_ids.extend(patch_ids)
#         structure_ids = torch.tensor(structure_ids, dtype=torch.long)

#         input_ids = pack_fim_inputs(
#             self.fim_tokens_ids, left_context_ids, right_context_ids, target_ids, self.structure_token_id, num_structure_tokens)

#         item = {"input_ids": input_ids, 'structure_ids': structure_ids, 'num_structure_tokens': num_structure_tokens}

#         return item


class AstCfcDataset(Dataset):
    """Dataset type: n chunks of cross-file context, m lines each, stored as an array
    """
    def __init__(self,
                 data,
                 training_stage,
                 ast_tokenizer,
                 code_tokenizer,
                 structure_token_id,
                 fim_tokens_ids,
                 num_structure_tokens=None,
                 max_seq_length=2048,
                 max_structure_length=512,
                 lc_rc_ratio=2.0):
        super(AstCfcDataset, self).__init__()
        self.data = data
        self.max_seq_length = max_seq_length
        self.code_tokenizer = code_tokenizer
        self.ast_tokenizer = ast_tokenizer
        self.structure_token_id = structure_token_id
        self.max_structure_length = max_structure_length
        self.lc_rc_ratio = lc_rc_ratio
        self.num_structure_tokens = num_structure_tokens
        self.fim_tokens_ids = fim_tokens_ids
        self.training_stage = training_stage

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind):
        if self.training_stage == 0:
            # instead of unraveling this into 10x bigger dataset, just choose a random and increase number of epochs
            chunk = random.choice(self.data[ind]['content']['crossfile_array'])
            # remove decommenting?
            cfc = '\n'.join(chunk.splitlines()[1:]).replace('#', '')  # removing file path in the first line and decommenting

            input_ids = self.code_tokenizer(cfc).input_ids  # turning cfc to tokens of the language model

            ast_tokens = AST(cfc, 'python', self.ast_tokenizer)
            ast_tokens = ast_tokens[:self.max_structure_length - 4]  # 4 special tokens for unixcoder
            chunk_tokens = [self.ast_tokenizer.cls_token, "<encoder-only>", self.ast_tokenizer.sep_token] \
                + ast_tokens + [self.ast_tokenizer.sep_token]
            structure_ids = self.ast_tokenizer.convert_tokens_to_ids(chunk_tokens)

            item = {"input_ids": input_ids, 'structure_ids': structure_ids}

        else:

            left_context_ids = self.code_tokenizer(self.data[ind]['content']['prompt'], return_tensors='pt').input_ids[0]
            right_context_ids = self.code_tokenizer(self.data[ind]['content']['right_context'], return_tensors='pt').input_ids[0]
            target_ids = self.code_tokenizer(self.data[ind]['content']['groundtruth'], return_tensors='pt').input_ids[0]

            tgt_len = len(target_ids)
            if not self.num_structure_tokens:
                self.num_structure_tokens = len(self.data[ind]['content']['crossfile_array'])
            lr_budget = self.max_seq_length - tgt_len - self.num_structure_tokens - 3  # 3 tokens for FIM
            rc_budget = int(lr_budget / (self.lc_rc_ratio + 1))
            lc_budget = int(rc_budget * self.lc_rc_ratio)

            left_context_ids = left_context_ids[-lc_budget:]
            right_context_ids = right_context_ids[:rc_budget]

            structure_ids = []
            for chunk in self.data[ind]['content']['crossfile_array'][:self.num_structure_tokens]:
                cfc = '\n'.join(chunk.splitlines()[1:])  # removing file path in the first line
                ast_tokens = AST(cfc.replace('#', ''), 'python', self.ast_tokenizer)  # decommenting
                ast_tokens = ast_tokens[:self.max_structure_length - 4]  # 4 special tokens for unixcoder
                chunk_tokens = [self.ast_tokenizer.cls_token, "<encoder-only>", self.ast_tokenizer.sep_token] \
                    + ast_tokens + [self.ast_tokenizer.sep_token]
                chunk_ids = self.ast_tokenizer.convert_tokens_to_ids(chunk_tokens)
                structure_ids.append(F.pad(torch.tensor(chunk_ids), (0, self.max_structure_length-len(chunk_ids)), value=self.ast_tokenizer.pad_token_id))
            structure_ids = torch.hstack(structure_ids)

            input_ids = pack_fim_inputs(
                self.fim_tokens_ids, left_context_ids, right_context_ids, target_ids, self.structure_token_id, self.num_structure_tokens)

            item = {"input_ids": input_ids, 'structure_ids': structure_ids, 'num_structure_tokens': self.num_structure_tokens}

        return item


class CodeAstCfcDataset(Dataset):
    def __init__(self,
                 data,
                 ast_tokenizer,
                 code_tokenizer,
                 structure_token_id,
                 max_seq_length=2048,
                 max_structure_length=512):
        super(CodeAstCfcDataset, self).__init__()
        self.data = data
        self.max_seq_length = max_seq_length
        self.code_tokenizer = code_tokenizer
        self.ast_tokenizer = ast_tokenizer
        self.structure_token_id = structure_token_id
        self.max_structure_length = max_structure_length

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

        cfc_tokens = []
        for token_id in cfc_ids:
            cfc_tokens.append(self.code_tokenizer.decode([token_id]))
        cfc = ''.join(cfc_tokens)

        # AST function ignores comments
        ast_tokens = AST(cfc.replace('#', ''), 'python', self.ast_tokenizer)
        structure_tokens = cfc_tokens + ast_tokens
        patch_length = self.max_structure_length - 4  # 4 special tokens for unixcoder
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
            fim_prefix,
            left_context_ids,
            fim_suffix,
            right_context_ids,
            torch.tensor([self.structure_token_id] * num_structure_tokens),
            fim_middle,
            target_ids]).to(torch.long)

        item = {"input_ids": input_ids, 'structure_ids': structure_ids, 'num_structure_tokens': num_structure_tokens}
        return item
