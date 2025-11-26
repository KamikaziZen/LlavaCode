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
                 num_truth_lines=10,
                 max_seq_length=2048,
                 max_structure_length=512,
                 lc_rc_ratio=2.0):
        super(Qwen3Dataset, self).__init__()
        # self.data = data
        print('Dataset samples before: ', len(data))
        # Examples with 0 cfc in TheStackv1
        remove_indices = [4689, 4690, 10037, 10998, 13381, 14865, 15364, 17490, 20118, 32910, 39973, 41641, 46718, 58023, 58643, 58856, 64421, 68036, 68990, 72104, 72105, 72690, 72691, 73997, 75598, 81033, 88847, 90688, 93281, 95307, 95500, 95606, 107740, 107742, 114358, 115832, 120194, 134935, 136458]
        # Examples with < 2 cfc in TheStackv2
        remove_indices += [1157732, 1157733, 1157734, 1157735, 1157736, 1157737, 1157738, 1157739, 1157740, 1157741, 1157742, 1157743, 1157744, 1157745, 1157746, 1157747, 1157748, 1157749, 1157750, 1157751, 1157752, 1157753, 1157754, 1157755, 1157756, 1157757, 1157758, 1157759, 1157760, 1157761, 1157762, 1157763, 1157764, 1157765, 1157766, 1157767, 1157768, 1157769, 1157770, 1157771, 1157772, 1157773, 1157774, 1157775, 1157776, 1157777, 1157778, 1157779, 1157780, 1157781, 1157782, 1157783, 1157784, 1157785, 1157786, 1157787, 1157788, 1157789, 1157790, 1157791, 1157792, 1157793, 1157794, 1157795, 1157796, 1157797, 1157798, 1157799, 1157800, 1157801, 1157802, 1157803, 1157804, 1157805, 1157806, 1157807, 1157808, 1157809, 1157810, 1157811, 1157812, 1157813, 1157814, 1157815, 1157816, 1157817, 1157818, 1157819, 1157820, 1157821, 1157822, 1157823, 1157824, 1157825, 1157826, 1157827, 1157828, 1157829, 1157830, 1157831]
        keep_indices = [i for i in range(len(data)) if i not in remove_indices]
        self.data = data.select(keep_indices)
        print('Dataset samples: ', len(self.data))
        self.max_seq_length = max_seq_length
        self.code_tokenizer = code_tokenizer
        self.structure_tokenizer = structure_tokenizer
        self.num_structure_tokens = num_structure_tokens
        self.structure_token_id = structure_token_id
        self.num_truth_lines = num_truth_lines
        self.max_structure_length = max_structure_length
        self.lc_rc_ratio = lc_rc_ratio
        self.fim_tokens_ids = fim_tokens_ids
        self.training_stage = training_stage
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

            chunk = self.data[orig_ind].get('content', self.data[orig_ind])['crossfile_array'][sub_ind]
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

            content = self.data[ind].get('content', self.data[ind])  # kostyl for stackv1 and stackv2 compatibility
            self.code_tokenizer.truncation_side = 'left'  # left context should be truncated from the left side
            left_context_ids = self.code_tokenizer(content['prompt'], return_tensors='pt', truncation=True, max_length=self.max_seq_length).input_ids[0]
            self.code_tokenizer.truncation_side = 'right'  # right context should be truncated from the right side
            right_context_ids = self.code_tokenizer(content['right_context'], return_tensors='pt', truncation=True, max_length=self.max_seq_length).input_ids[0]
            groundtruth = "\n".join(content['groundtruth'].split("\n")[:self.num_truth_lines])
            target_ids = self.code_tokenizer(groundtruth, return_tensors='pt', truncation=True, max_length=self.max_seq_length).input_ids[0]

            # tgt_len = len(target_ids)
            lr_budget = self.max_seq_length - 50 - self.num_structure_tokens - 3  # 3 tokens for FIM, 50 for line completion
            rc_budget = int(lr_budget / (self.lc_rc_ratio + 1))
            lc_budget = int(rc_budget * self.lc_rc_ratio)

            left_context_ids = left_context_ids[-lc_budget:]
            right_context_ids = right_context_ids[:rc_budget]

            num_structure_tokens = min(self.num_structure_tokens, len(content['crossfile_array']))
            structure_ids = torch.empty(0, dtype=torch.long)
            for chunk in content['crossfile_array'][:num_structure_tokens]:
                cfc = '\n'.join(chunk.splitlines()[1:])  # removing file path in the first line
                cfc_ids = self.structure_tokenizer(cfc, return_tensors='pt', truncation=True, max_length=self.max_structure_length).input_ids[0]
                structure_ids = torch.hstack([structure_ids, F.pad(cfc_ids, (0, self.max_structure_length-len(cfc_ids)), value=self.structure_tokenizer.pad_token_id)])

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
