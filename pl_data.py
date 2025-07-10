import logging
import time

from datasets import load_from_disk
from lightning.pytorch import LightningDataModule
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import math
from sklearn.preprocessing import LabelEncoder

from preprocess import AST
from  utils import code_to_graph

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# NODE_TYPES = [
#     'Add', 'And', 'AnnAssign', 'Assert', 'Assign', 'AsyncFor',
#     'AsyncFunctionDef', 'AsyncWith', 'Attribute', 'AugAssign', 'Await',
#     'BinOp', 'BitAnd', 'BitOr', 'BitXor', 'BoolOp', 'Break', 'Call',
#     'ClassDef', 'Compare', 'Constant', 'Continue', 'Del', 'Delete',
#     'Dict', 'DictComp', 'Div', 'Eq', 'ExceptHandler', 'Expr',
#     'FloorDiv', 'For', 'FormattedValue', 'FunctionDef', 'GeneratorExp',
#     'Global', 'Gt', 'GtE', 'If', 'IfExp', 'Import', 'ImportFrom', 'In',
#     'Invert', 'Is', 'IsNot', 'JoinedStr', 'LShift', 'Lambda', 'List',
#     'ListComp', 'Load', 'Lt', 'LtE', 'MatMult', 'Mod', 'Module',
#     'Mult', 'Name', 'Nonlocal', 'Not', 'NotEq', 'NotIn', 'Or', 'Pass',
#     'Pow', 'RShift', 'Raise', 'Return', 'Set', 'SetComp', 'Slice',
#     'Starred', 'Store', 'Sub', 'Subscript', 'Try', 'Tuple', 'UAdd',
#     'USub', 'UnaryOp', 'While', 'With', 'Yield', 'YieldFrom', 'alias',
#     'arg', 'arguments', 'comprehension', 'keyword', 'withitem'
# ]


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


class CodeCfcDataset_old(Dataset):
    """Dataset type: 3 chunks of cross-file context, 40 lines each, merged into a single string
    """
    def __init__(self,
                 data,
                 ast_tokenizer,
                 code_tokenizer,
                 structure_token_id,
                 max_seq_length=2048,
                 max_structure_length=512):
        super(CodeCfcDataset, self).__init__()
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

        structure_tokens = self.ast_tokenizer.tokenize(
            self.code_tokenizer.decode(self.data[ind]['cfc_token_ids']))
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


class CodeCfcDataset(Dataset):
    """Dataset type: 10 chunks of cross-file context, 10 lines each, stored as an array
    """
    def __init__(self,
                 data,
                 ast_tokenizer,
                 code_tokenizer,
                 structure_token_id,
                 max_seq_length=2048,
                 max_structure_length=512,
                 lc_rc_ratio=2.0):
        super(CodeCfcDataset, self).__init__()
        self.data = data
        self.max_seq_length = max_seq_length
        self.code_tokenizer = code_tokenizer
        self.ast_tokenizer = ast_tokenizer
        self.structure_token_id = structure_token_id
        self.max_structure_length = max_structure_length
        self.lc_rc_ratio = lc_rc_ratio

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
        num_structure_tokens = len(self.data[ind]['content']['crossfile_array'])
        lr_budget = self.max_seq_length - tgt_len - num_structure_tokens - 3  # 3 tokens for FIM
        rc_budget = int(lr_budget / (self.lc_rc_ratio + 1))
        lc_budget = int(rc_budget * self.lc_rc_ratio)

        left_context_ids = left_context_ids[-lc_budget:]
        right_context_ids = right_context_ids[:rc_budget]

        structure_ids = []
        for chunk in self.data[ind]['content']['crossfile_array']:
            cfc = '\n'.join(chunk.splitlines()[1:])  # removing file path in the first line
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
            torch.tensor([self.structure_token_id] * num_structure_tokens),
            fim_middle_id,
            target_ids]).to(torch.long)

        item = {"input_ids": input_ids, 'structure_ids': structure_ids, 'num_structure_tokens': num_structure_tokens}
        return item


class AstCfcDataset_old(Dataset):
    """Dataset type: 3 chunks of cross-file context, 40 lines each, merged into a single string
    """
    def __init__(self,
                 data,
                 ast_tokenizer,
                 code_tokenizer,
                 structure_token_id,
                 max_seq_length=2048,
                 max_structure_length=512):
        super(AstCfcDataset, self).__init__()
        self.data = data
        self.max_seq_length = max_seq_length
        self.code_tokenizer = code_tokenizer
        self.ast_tokenizer = ast_tokenizer
        self.structure_token_id = structure_token_id
        self.max_structure_length = max_structure_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind):
        fim_prefix, fim_suffix, fim_middle = torch.tensor([1]), torch.tensor([3]), torch.tensor([2])
        left_context_ids = torch.tensor(self.data[ind]['lc_token_ids'])
        right_context_ids = torch.tensor(self.data[ind]['rc_token_ids'])
        cfc_ids = torch.tensor(self.data[ind]['cfc_token_ids'])
        target_ids = torch.tensor(self.data[ind]['tgt_token_ids'])

        cfc = self.code_tokenizer.decode(cfc_ids)
        # AST function ignores comments
        ast_tokens = AST(cfc.replace('#', ''), 'python', self.ast_tokenizer)
        patch_length = self.max_structure_length - 4  # 4 special tokens for unixcoder
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
            fim_prefix,
            left_context_ids,
            fim_suffix,
            right_context_ids,
            torch.tensor([self.structure_token_id] * num_structure_tokens),
            fim_middle,
            target_ids]).to(torch.long)

        item = {"input_ids": input_ids, 'structure_ids': structure_ids, 'num_structure_tokens': num_structure_tokens}
        return item


class AstCfcDataset(Dataset):
    """Dataset type: 10 chunks of cross-file context, 10 lines each, stored as an array
    """
    def __init__(self,
                 data,
                 ast_tokenizer,
                 code_tokenizer,
                 structure_token_id,
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
        num_structure_tokens = len(self.data[ind]['content']['crossfile_array'])
        lr_budget = self.max_seq_length - tgt_len - num_structure_tokens - 3  # 3 tokens for FIM
        rc_budget = int(lr_budget / (self.lc_rc_ratio + 1))
        lc_budget = int(rc_budget * self.lc_rc_ratio)

        left_context_ids = left_context_ids[-lc_budget:]
        right_context_ids = right_context_ids[:rc_budget]

        structure_ids = []
        for chunk in self.data[ind]['content']['crossfile_array']:
            cfc = '\n'.join(chunk.splitlines()[1:])  # removing file path in the first line
            ast_tokens = AST(cfc.replace('#', ''), 'python', self.ast_tokenizer)  # decommenting
            ast_tokens = ast_tokens[:self.max_structure_length - 4]  # 4 special tokens for unixcoder
            chunk_tokens = [self.ast_tokenizer.cls_token, "<encoder-only>", self.ast_tokenizer.sep_token] \
                + ast_tokens + [self.ast_tokenizer.sep_token]
            chunk_ids = self.ast_tokenizer.convert_tokens_to_ids(chunk_tokens)
            structure_ids.extend(F.pad(torch.tensor(chunk_ids), (0, self.max_structure_length-len(chunk_ids)), value=self.ast_tokenizer.pad_token_id))
        structure_ids = torch.tensor(structure_ids, dtype=torch.long)

        input_ids = torch.cat([
            fim_prefix_id,
            left_context_ids,
            fim_suffix_id,
            right_context_ids,
            torch.tensor([self.structure_token_id] * num_structure_tokens),
            fim_middle_id,
            target_ids]).to(torch.long)

        item = {"input_ids": input_ids, 'structure_ids': structure_ids, 'num_structure_tokens': num_structure_tokens}
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


# class GraphCfcDataset(Dataset):
#     def __init__(self,
#                  data,
#                  code_tokenizer,
#                  structure_token_id,
#                  max_seq_length=2048,
#                  max_structure_length=512):
#         super(GraphCfcDataset, self).__init__()
#         self.data = data
#         self.max_seq_length = max_seq_length
#         self.code_tokenizer = code_tokenizer
#         self.structure_token_id = structure_token_id
#         self.max_structure_length = max_structure_length

#         self.node_type_encoder = LabelEncoder().fit(NODE_TYPES)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, ind):
#         # indexing the chunked data directly
#         # source_tokens = torch.tensor(self.data[ind]['token_ids'])
#         fim_prefix, fim_suffix, fim_middle = torch.tensor([1]), torch.tensor([3]), torch.tensor([2])
#         left_context_ids = torch.tensor(self.data[ind]['lc_token_ids'])
#         right_context_ids = torch.tensor(self.data[ind]['rc_token_ids'])
#         cfc_ids = torch.tensor(self.data[ind]['cfc_token_ids'])
#         target_ids = torch.tensor(self.data[ind]['tgt_token_ids'])

#         cfc = self.code_tokenizer.decode(cfc_ids)
#         graph_data = code_to_graph(cfc)  # cfc is usually hidden in comments
#         if graph_data:
#             graph, node_features = graph_data['graph'], graph_data['node_features']
#             num_nodes = len(node_features)
#             edges_ids = torch.tensor(list(graph.edges()), dtype=torch.long).t().contiguous()

#             nodes_ids = torch.zeros((num_nodes), dtype=torch.long)
#             for idx, features in node_features.items():
#                 nodes_ids[idx] = self.node_type_encoder.transform([features['type']])[0]

#             num_structure_tokens = math.ceil(len(nodes_ids) / self.max_structure_length)

#         else:
#             print('Failed to extract graph from ')
#             print(cfc)
#             print('-------')
#             num_structure_tokens = 0
#             nodes_ids = None
#             edges_ids = None

#         input_ids = torch.cat([
#             fim_prefix,
#             left_context_ids,
#             fim_suffix,
#             right_context_ids,
#             torch.tensor([self.structure_token_id] * num_structure_tokens),
#             fim_middle,
#             target_ids]).to(torch.long)

#         item = {'input_ids': input_ids, 'structure_ids': nodes_ids,
#                 'edges_ids': edges_ids, 'num_structure_tokens': num_structure_tokens, 'code': cfc}
#         return item


class LlavaCodeDataCollator:
    def __init__(self, code_tokenizer, ast_tokenizer):
        self.code_tokenizer = code_tokenizer
        self.ast_tokenizer = ast_tokenizer

    def __call__(self, features):
        input_ids = [{'input_ids': f['input_ids']} for f in features]
        structure_ids = [{'input_ids': f['structure_ids']} for f in features]

        batch = self.code_tokenizer.pad(
            input_ids,
            padding=True,
            return_tensors='pt',
            padding_side='right'
        )

        structure_batch = self.ast_tokenizer.pad(
            structure_ids,
            padding=True,
            pad_to_multiple_of=512,
            return_tensors='pt',
            padding_side='right'
        )

        batch['structure_ids'] = structure_batch['input_ids']
        batch['num_structure_tokens'] = torch.tensor([f['num_structure_tokens'] for f in features], dtype=torch.int)

        return batch


class DataModule(LightningDataModule):
    def __init__(self, data_prefix, train_datadir, valid_datadir, train_batch_size,
                 valid_batch_size, code_tokenizer, ast_tokenizer, structure_token_id,
                 num_workers=0,):
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
            return AstLcontextDataset(
                raw_data,
                code_tokenizer=self.code_tokenizer,
                ast_tokenizer=self.ast_tokenizer,
                structure_token_id=self.structure_token_id,
                max_structure_length=512)
        elif self.data_prefix == 'ast_cfc':
            return AstCfcDataset(
                raw_data,
                code_tokenizer=self.code_tokenizer,
                ast_tokenizer=self.ast_tokenizer,
                structure_token_id=self.structure_token_id,
                max_structure_length=512)
        elif self.data_prefix == 'code_cfc':
            return CodeCfcDataset(
                raw_data,
                code_tokenizer=self.code_tokenizer,
                ast_tokenizer=self.ast_tokenizer,
                structure_token_id=self.structure_token_id,
                max_structure_length=512)
        elif self.data_prefix == 'codeast_cfc':
            return CodeAstCfcDataset(
                raw_data,
                code_tokenizer=self.code_tokenizer,
                ast_tokenizer=self.ast_tokenizer,
                structure_token_id=self.structure_token_id,
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
