import logging

import torch
from torch.utils.data import Dataset

from parser import (
    DFG_python,
    DFG_java,
)
from parser import (
    remove_comments_and_docstrings,
    tree_to_token_index,
    index_to_code_token
)
from tree_sitter import Language, Parser

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


dfg_function = {
    'python': DFG_python,
    'java': DFG_java,
    # 'ruby':DFG_ruby,
    # 'go':DFG_go,
    # 'php':DFG_php,
    # 'javascript':DFG_javascript
}

# load parsers
parsers = {}
for lang in dfg_function:
    LANGUAGE = Language('parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE) 
    parser = [parser, dfg_function[lang]]
    parsers[lang] = parser


# remove comments, tokenize code and extract dataflow
def extract_dataflow(code, parser, lang):
    # remove comments
    try:
        code = remove_comments_and_docstrings(code, lang)
    except Exception as e:
        pass

    # obtain dataflow
    if lang == "php":
        code = "<?php"+code+"?>"
    try:
        tree = parser[0].parse(bytes(code, 'utf8'))
        root_node = tree.root_node
        tokens_index = tree_to_token_index(root_node)
        code = code.split('\n')
        code_tokens = [index_to_code_token(x, code) for x in tokens_index]
        index_to_code = {}
        for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
            index_to_code[index] = (idx, code)
        try:
            DFG, _ = parser[1](root_node, index_to_code, {})
        except Exception as e:
            logger.warning(e)
            DFG = []
        DFG = sorted(DFG, key=lambda x: x[1])
        idxs = set()
        for d in DFG:
            if len(d[-1]) != 0:
                idxs.add(d[1])
            for x in d[-1]:
                idxs.add(x)
        new_DFG = []
        for d in DFG:
            if d[1] in idxs:
                new_DFG.append(d)
        dfg = new_DFG
    except Exception as e:
        dfg = []

    return code_tokens, dfg


class DfgDataset(Dataset):
    """
    """
    def __init__(self,
                 data,
                 dfg_tokenizer,
                 code_tokenizer,
                 structure_token_id,
                 num_structure_tokens=None,
                 max_seq_length=2048,
                 max_structure_length=512,
                 lc_rc_ratio=2.0):
        super(DfgDataset, self).__init__()
        self.data = data
        self.max_seq_length = max_seq_length
        self.code_tokenizer = code_tokenizer
        self.dfg_tokenizer = dfg_tokenizer
        self.structure_token_id = structure_token_id
        self.max_structure_length = max_structure_length
        self.lc_rc_ratio = lc_rc_ratio
        self.num_structure_tokens = num_structure_tokens

        self.parser = parsers['python']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind):
        fim_prefix_id = torch.tensor(self.code_tokenizer.convert_tokens_to_ids(['<fim_prefix>']))
        fim_suffix_id = torch.tensor(self.code_tokenizer.convert_tokens_to_ids(['<fim_suffix>']))
        fim_middle_id = torch.tensor(self.code_tokenizer.convert_tokens_to_ids(['<fim_middle>']))

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
        position_idxs = []
        attn_masks = []
        for chunk in self.data[ind]['content']['crossfile_array'][:self.num_structure_tokens]:
            cfc = '\n'.join(chunk.splitlines()[1:])  # removing file path in the first line
            # extract data flow
            code_tokens, dfg = extract_dataflow(cfc, parser, 'python')
            code_tokens = [self.dfg_tokenizer.tokenize('@ '+x)[1:] if idx != 0 else self.dfg_tokenizer.tokenize(x) for idx, x in enumerate(code_tokens)]
            ori2cur_pos = {}
            ori2cur_pos[-1] = (0, 0)
            for i in range(len(code_tokens)):
                ori2cur_pos[i] = (ori2cur_pos[i-1][1], ori2cur_pos[i-1][1]+len(code_tokens[i]))
            # turning list of lists into a list, same as flatten()
            code_tokens = [y for x in code_tokens for y in x]

            # truncating
            # TODO: look up better value than 256
            code_tokens = code_tokens[:self.max_structure_length-3-min(len(dfg), 256)][:512-3]  # 3 special tokens of GraphCodeBert
            source_tokens = [self.dfg_tokenizer.cls_token] + code_tokens + [self.dfg_tokenizer.sep_token]
            source_ids = self.dfg_tokenizer.convert_tokens_to_ids(source_tokens)
            position_idx = [i+self.dfg_tokenizer.pad_token_id + 1 for i in range(len(source_tokens))]
            dfg = dfg[:self.max_structure_length-len(source_tokens)]
            source_tokens += [x[0] for x in dfg]
            position_idx += [0 for x in dfg]
            source_ids += [self.dfg_tokenizer.unk_token_id for x in dfg]
            padding_length = self.max_structure_length - len(source_ids)
            position_idx += [self.dfg_tokenizer.pad_token_id] * padding_length
            source_ids += [self.dfg_tokenizer.pad_token_id] * padding_length

            # reindex
            reverse_index = {}
            for idx, x in enumerate(dfg):
                reverse_index[x[1]] = idx
            for idx, x in enumerate(dfg):
                dfg[idx] = x[:-1]+([reverse_index[i] for i in x[-1] if i in reverse_index],)
            dfg_to_dfg = [x[-1] for x in dfg]
            dfg_to_code = [ori2cur_pos[x[1]] for x in dfg]
            length = len([self.dfg_tokenizer.cls_token])
            dfg_to_code = [(x[0]+length, x[1]+length) for x in dfg_to_code]

            # calculate graph-guided masked function
            attn_mask = torch.zeros((self.max_structure_length, self.max_structure_length), dtype=bool)

            # calculate begin index of node and max length of input
            node_index = sum([i > 1 for i in position_idx])
            max_length = sum([i != 1 for i in position_idx])
            # sequence can attend to sequence
            attn_mask[:node_index, :node_index] = True
            # special tokens attend to all tokens
            for idx, i in enumerate(source_ids):
                if i in [0, 2]:
                    attn_mask[idx, :max_length] = True
            # nodes attend to code tokens that are identified from
            for idx, (a, b) in enumerate(dfg_to_code):
                if a < node_index and b < node_index:
                    attn_mask[idx+node_index, a:b] = True
                    attn_mask[a:b, idx+node_index] = True
            # nodes attend to adjacent nodes
            for idx, nodes in enumerate(dfg_to_dfg):
                for a in nodes:
                    if a+node_index < len(position_idx):
                        attn_mask[idx+node_index, a+node_index] = True

            structure_ids.extend(source_ids)
            position_idxs.extend(position_idx)
            attn_masks.append(attn_mask.flatten())

        structure_ids = torch.tensor(structure_ids, dtype=torch.long)
        position_idxs = torch.tensor(position_idxs, dtype=torch.long)
        attn_masks = torch.hstack(attn_masks)

        input_ids = torch.cat([
            fim_prefix_id,
            left_context_ids,
            fim_suffix_id,
            right_context_ids,
            torch.tensor([self.structure_token_id] * self.num_structure_tokens),
            fim_middle_id,
            target_ids]).to(torch.long)

        item = {'input_ids': input_ids, 'structure_ids': structure_ids,
                'structure_pos_idx': position_idxs, 'structure_attn_mask': attn_masks,
                'num_structure_tokens': self.num_structure_tokens}
        return item
