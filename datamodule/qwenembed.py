import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset

import random

from .const import PARAPHRASE_CUES, STRUCTURE_TOKEN


class Qwen3Dataset(Dataset):
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
        super(Qwen3Dataset, self).__init__()
        # self.data = data
        print('Dataset samples before: ', len(data))
        remove_indices = [4689, 4690, 10037, 10998, 13381, 14865, 15364, 17490, 20118, 32910, 39973, 41641, 46718, 58023, 58643, 58856, 64421, 68036, 68990, 72104, 72105, 72690, 72691, 73997, 75598, 81033, 88847, 90688, 93281, 95307, 95500, 95606, 107740, 107742, 114358, 115832, 120194, 134935, 136458]
        keep_indices = [i for i in range(len(data)) if i not in remove_indices]
        self.data = data.select(keep_indices)
        print('Dataset samples: ', len(self.data))
        self.max_seq_length = max_seq_length
        self.code_tokenizer = code_tokenizer
        self.structure_tokenizer = structure_tokenizer
        self.num_structure_tokens = num_structure_tokens
        self.structure_token_id = structure_token_id
        self.max_structure_length = max_structure_length
        self.lc_rc_ratio = lc_rc_ratio
        self.fim_tokens_ids = fim_tokens_ids
        self.training_stage = training_stage
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

            chunk = self.data[orig_ind]['content']['crossfile_array'][sub_ind]
            cfc = '\n'.join(chunk.splitlines()[1:])  # removing file path in the first line
            cfc_ids = self.structure_tokenizer(cfc, return_tensors='pt', truncation=True, max_length=self.max_structure_length).input_ids[0]

            target_ids = self.code_tokenizer(cfc, return_tensors='pt', truncation=True, max_length=self.max_seq_length).input_ids[0]

            input_ids = torch.cat([
                fim_prefix_id,
                torch.tensor([self.structure_token_id]),
                self.code_tokenizer(random.choice(PARAPHRASE_CUES), return_tensors='pt').input_ids[0],
                # fim_suffix_id,
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
                cfc_ids = self.structure_tokenizer(cfc, return_tensors='pt', truncation=True, max_length=self.max_structure_length).input_ids[0]

                structure_ids.extend(F.pad(cfc_ids, (0, self.max_structure_length-len(cfc_ids)), value=self.structure_tokenizer.pad_token_id))
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
