import logging
import time

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import math
import random
from preprocess import AST

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
                 max_seq_length=2100,
                 max_structure_length=512,
                 lc_rc_ratio=2.0):
        super(CodeCfcDataset, self).__init__()
        # self.data = data
        print('Dataset samples before: ', len(data))
        # samples with less then 10 cross-file contexts
        remove_indices = [4689, 4690, 10037, 10998, 13381, 14865, 15364, 17490, 20118, 32910, 39973, 41641, 46718, 58023, 58643, 58856, 64421, 68036, 68990, 72104, 72105, 72690, 72691, 73997, 75598, 81033, 88847, 90688, 93281, 95307, 95500, 95606, 107740, 107742, 114358, 115832, 120194, 134935, 136458]
        keep_indices = [i for i in range(len(data)) if i not in remove_indices]
        self.data = data.select(keep_indices)
        print('Dataset samples: ', len(self.data))
        self.max_seq_length = max_seq_length
        self.max_target_length = 100
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
            self.expand_factor = len(self.data[0].get('content', self.data[0])['crossfile_array'])

    def __len__(self):

        return len(self.data) * self.expand_factor

    def __getitem__(self, ind):

        fim_prefix_id, fim_suffix_id, fim_middle_id = self.fim_tokens_ids.reshape(3, 1)

        if self.training_stage == 0:

            orig_ind = ind // self.expand_factor
            sub_ind = ind % self.expand_factor

            cfc = self.data[orig_ind].get('content', self.data[orig_ind])['crossfile_array'][sub_ind]
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

            content = self.data[ind].get('content', self.data[ind])  # kostyl for stackv1 and stackv2 compatibility
            self.code_tokenizer.truncation_side = 'left'  # left context should be truncated from the left side
            left_context_ids = self.code_tokenizer(content['prompt'], return_tensors='pt', truncation=True, max_length=self.max_seq_length).input_ids[0]
            self.code_tokenizer.truncation_side = 'right'  # right context should be truncated from the right side
            right_context_ids = self.code_tokenizer(content['right_context'], return_tensors='pt', truncation=True, max_length=self.max_seq_length).input_ids[0]
            groundtruth = "\n".join(content['groundtruth'].split("\n")[:self.num_truth_lines])
            target_ids = self.code_tokenizer(groundtruth, return_tensors='pt', truncation=True, max_length=self.max_target_length).input_ids[0]

            lr_budget = self.max_seq_length - self.max_target_length - self.num_structure_tokens - 3  # 3 tokens for FIM + target budget
            rc_budget = int(lr_budget / (self.lc_rc_ratio + 1))
            lc_budget = int(rc_budget * self.lc_rc_ratio)

            left_context_ids = left_context_ids[-lc_budget:]
            right_context_ids = right_context_ids[:rc_budget]

            num_structure_tokens = min(self.num_structure_tokens, len(content['crossfile_array']))
            structure_ids = torch.empty(0, dtype=torch.long)
            for chunk in content['crossfile_array'][:num_structure_tokens]:
                cfc = '\n'.join(chunk.splitlines()[1:])  # removing file path in the first line
                code_tokens = self.ast_tokenizer.tokenize(cfc)  # TODO: try decommenting?
                code_tokens = code_tokens[:self.max_structure_length - 4]  # 4 special tokens for unixcoder
                chunk_tokens = [self.ast_tokenizer.cls_token, "<encoder-only>", self.ast_tokenizer.sep_token] \
                    + code_tokens + [self.ast_tokenizer.sep_token]
                chunk_ids = self.ast_tokenizer.convert_tokens_to_ids(chunk_tokens)
                structure_ids = torch.hstack([structure_ids, F.pad(torch.tensor(chunk_ids), (0, self.max_structure_length-len(chunk_ids)), value=self.ast_tokenizer.pad_token_id)])

            input_ids = torch.cat([
                torch.tensor([self.structure_token_id] * num_structure_tokens),
                fim_prefix_id,
                left_context_ids,
                fim_suffix_id,
                right_context_ids,
                # self.code_tokenizer('\n# Here are some relevant code fragments from other files of the repo:', return_tensors='pt').input_ids[0],
                # torch.tensor([self.structure_token_id] * self.num_structure_tokens),
                fim_middle_id,
                target_ids]).to(torch.long)

            # for KL-div training
            all_cfc = '\n'.join(content['crossfile_array'][:num_structure_tokens])
            all_cfc = '\n# Here are some relevant code fragments from other files of the repo:\n' + all_cfc
            all_cfc_ids = self.code_tokenizer(all_cfc, return_tensors='pt', truncation=True, max_length=self.max_seq_length).input_ids[0]
            teacher_input_ids = torch.cat([
                all_cfc_ids,
                fim_prefix_id,
                left_context_ids,
                fim_suffix_id,
                right_context_ids,
                # all_cfc_ids,
                fim_middle_id,
                target_ids]).to(torch.long)

            item = {"input_ids": input_ids, 'teacher_input_ids': teacher_input_ids, 'structure_ids': structure_ids, 'num_structure_tokens': num_structure_tokens}

        return item


class AstCfcDataset(Dataset):
    """Dataset type: n chunks of cross-file context, m lines each, stored as an array
    """
    def __init__(self,
                 data,
                 training_stage,
                 language,
                 ast_tokenizer,
                 code_tokenizer,
                 structure_token_id,
                 fim_tokens_ids,
                 num_truth_lines=10,
                 num_structure_tokens=None,
                 max_seq_length=2100,
                 max_structure_length=512,
                 lc_rc_ratio=2.0):
        super(AstCfcDataset, self).__init__()
        self.data = data
        print('Dataset samples before removal: ', len(data))
        # samples with less then 10 cross-file contexts
        remove_indices = [4689, 4690, 10037, 10998, 13381, 14865, 15364, 17490, 20118, 32910, 39973, 41641, 46718, 58023, 58643, 58856, 64421, 68036, 68990, 72104, 72105, 72690, 72691, 73997, 75598, 81033, 88847, 90688, 93281, 95307, 95500, 95606, 107740, 107742, 114358, 115832, 120194, 134935, 136458]
        # sample with circular AST in cross-file context
        remove_indices += [93399]
        keep_indices = [i for i in range(len(data)) if i not in remove_indices]
        self.data = data.select(keep_indices)
        print('Dataset samples after removal: ', len(self.data))
        self.max_seq_length = max_seq_length
        self.max_target_length = 100
        self.code_tokenizer = code_tokenizer
        self.ast_tokenizer = ast_tokenizer
        self.structure_token_id = structure_token_id
        self.max_structure_length = max_structure_length
        self.lc_rc_ratio = lc_rc_ratio
        self.num_truth_lines = num_truth_lines
        self.num_structure_tokens = num_structure_tokens
        self.fim_tokens_ids = fim_tokens_ids
        self.training_stage = training_stage
        self.language = language

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind):

        fim_prefix_id, fim_suffix_id, fim_middle_id = self.fim_tokens_ids.reshape(3, 1)

        if self.training_stage == 0:
            # instead of unraveling this into 10x bigger dataset, just choose a random and increase number of epochs
            chunk = random.choice(self.data[ind].get('content', self.data[ind])['crossfile_array'])
            # remove decommenting?
            cfc = '\n'.join(chunk.splitlines()[1:]).replace('#', '')  # removing file path in the first line and decommenting

            input_ids = self.code_tokenizer(cfc).input_ids  # turning cfc to tokens of the language model

            ast_tokens = AST(cfc, self.language, self.ast_tokenizer)
            ast_tokens = ast_tokens[:self.max_structure_length - 4]  # 4 special tokens for unixcoder
            chunk_tokens = [self.ast_tokenizer.cls_token, "<encoder-only>", self.ast_tokenizer.sep_token] \
                + ast_tokens + [self.ast_tokenizer.sep_token]
            structure_ids = self.ast_tokenizer.convert_tokens_to_ids(chunk_tokens)

            item = {"input_ids": input_ids, 'structure_ids': structure_ids}

        else:

            content = self.data[ind].get('content', self.data[ind])  # kostyl for stackv1 and stackv2 compatibility
            self.code_tokenizer.truncation_side = 'left'  # left context should be truncated from the left side
            left_context_ids = self.code_tokenizer(content['prompt'], return_tensors='pt', truncation=True, max_length=self.max_seq_length).input_ids[0]
            self.code_tokenizer.truncation_side = 'right'  # right context should be truncated from the right side
            right_context_ids = self.code_tokenizer(content['right_context'], return_tensors='pt', truncation=True, max_length=self.max_seq_length).input_ids[0]
            groundtruth = "\n".join(content['groundtruth'].split("\n")[:self.num_truth_lines])
            target_ids = self.code_tokenizer(groundtruth, return_tensors='pt', truncation=True, max_length=self.max_target_length).input_ids[0]

            lr_budget = self.max_seq_length - self.max_target_length - self.num_structure_tokens - 3  # 3 tokens for FIM + target budget
            rc_budget = int(lr_budget / (self.lc_rc_ratio + 1))
            lc_budget = int(rc_budget * self.lc_rc_ratio)

            left_context_ids = left_context_ids[-lc_budget:]
            right_context_ids = right_context_ids[:rc_budget]

            num_structure_tokens = min(self.num_structure_tokens, len(content['crossfile_array']))
            structure_ids = torch.empty(0, dtype=torch.long)
            for chunk in content['crossfile_array'][:num_structure_tokens]:
                cfc = '\n'.join(chunk.splitlines()[1:])  # removing file path in the first line
                ast_tokens = AST(cfc.replace('#', ''), self.language, self.ast_tokenizer)  # AST is not calculated for comments
                ast_tokens = ast_tokens[:self.max_structure_length - 4]  # 4 special tokens for unixcoder
                chunk_tokens = [self.ast_tokenizer.cls_token, "<encoder-only>", self.ast_tokenizer.sep_token] \
                    + ast_tokens + [self.ast_tokenizer.sep_token]
                chunk_ids = self.ast_tokenizer.convert_tokens_to_ids(chunk_tokens)
                structure_ids = torch.hstack([structure_ids, F.pad(torch.tensor(chunk_ids), (0, self.max_structure_length-len(chunk_ids)), value=self.ast_tokenizer.pad_token_id)])

            input_ids = torch.cat([
                torch.tensor([self.structure_token_id] * num_structure_tokens),
                fim_prefix_id,
                left_context_ids,
                fim_suffix_id,
                right_context_ids,
                # self.code_tokenizer('\n# Here are some relevant code fragments from other files of the repo:', return_tensors='pt').input_ids[0],
                # torch.tensor([self.structure_token_id] * num_structure_tokens),
                fim_middle_id,
                target_ids]).to(torch.long)

            # for KL-div training
            all_cfc = '\n'.join(content['crossfile_array'][:num_structure_tokens])
            all_cfc = '\n# Here are some relevant code fragments from other files of the repo:\n' + all_cfc
            all_cfc_ids = self.code_tokenizer(all_cfc, return_tensors='pt', truncation=True, max_length=self.max_seq_length).input_ids[0]
            teacher_input_ids = torch.cat([
                all_cfc_ids,
                fim_prefix_id,
                left_context_ids,
                fim_suffix_id,
                right_context_ids,
                # all_cfc_ids,
                fim_middle_id,
                target_ids]).to(torch.long)

            item = {"input_ids": input_ids, 'teacher_input_ids': teacher_input_ids, 'structure_ids': structure_ids, 'num_structure_tokens': num_structure_tokens}

        return item
