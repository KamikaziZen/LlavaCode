
import os
# os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    PretrainedConfig,
    PreTrainedModel,
    CONFIG_MAPPING,
    BitsAndBytesConfig,
    GenerationMixin,
)
from transformers.modeling_outputs import ModelOutput
from transformers.activations import ACT2FN
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from transformers.cache_utils import DynamicCache
from transformers.utils import can_return_tuple, LossKwargs
from transformers.processing_utils import Unpack

from tree_sitter import Language, Parser
import editdistance
import numpy as np
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from functools import partial
import json
import re
import gc
from tqdm import tqdm
from io import StringIO
import tokenize

STRUCTURE_TOKEN = '<CODE_STRUCTURE>'

FIMMAP = {
    'qwen2.5': ['<|fim_prefix|>', '<|fim_suffix|>', '<|fim_middle|>'],
    'starcoder': ['<fim_prefix>', '<fim_suffix>', '<fim_middle>'],
}

device = torch.device('cuda:1')


@dataclass
class ArgsMock:
    text_model_id = "Qwen/Qwen2.5-Coder-7B"
    structure_model_id = "microsoft/unixcoder-base"
    language = "java"
    task = "line_completion"
    data_prefix = "ast_cfc"
    num_structure_tokens = 5
    cfc_place = "preprefix"
    output_dir = "test_inference"
    projector = "3L"
    do_sample = False
    max_structure_length = 512
    max_seq_length = 2000
    gen_length = 50
    right_context_length = 512
    lc_rc_ratio = 2.0
    prompt_file = "test_example.jsonl"
    model_checkpoint = None
    projector_checkpoint = "./ckpt/qwen7_unixcoder_3l_java_2emes_stack2_projector.pth"


def remove_comments_and_docstrings(source, lang):
    if lang in ['python']:
        """
        Returns 'source' minus comments and docstrings.
        """
        io_obj = StringIO(source)
        out = ""
        prev_toktype = tokenize.INDENT
        last_lineno = -1
        last_col = 0
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type = tok[0]
            token_string = tok[1]
            start_line, start_col = tok[2]
            end_line, end_col = tok[3]
            ltext = tok[4]
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                out += (" " * (start_col - last_col))
            # Remove comments:
            if token_type == tokenize.COMMENT:
                pass
            # This series of conditionals removes docstrings:
            elif token_type == tokenize.STRING:
                if prev_toktype != tokenize.INDENT:
                    # This is likely a docstring; double-check we're not inside an operator:
                    if prev_toktype != tokenize.NEWLINE:
                        if start_col > 0:
                            out += token_string
            else:
                out += token_string
            prev_toktype = token_type
            last_col = end_col
            last_lineno = end_line
        temp = []
        for x in out.split('\n'):
            if x.strip() != "":
                temp.append(x)
        return '\n'.join(temp)
    elif lang in ['ruby']:
        return source
    else:
        def replacer(match):
            s = match.group(0)
            if s.startswith('/'):
                return " "  # note: a space and not an empty string
            else:
                return s
        pattern = re.compile(
            r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
            re.DOTALL | re.MULTILINE
        )
        temp = []
        for x in re.sub(pattern, replacer, source).split('\n'):
            if x.strip() != "":
                temp.append(x)
        return '\n'.join(temp)


def tree_to_token_index(root_node):
    if (len(root_node.children) == 0 or root_node.type == 'string' or root_node.type == 'comment' or 'comment' in root_node.type):
        return [(root_node.start_point, root_node.end_point)]
    else:
        code_tokens = []
        for child in root_node.children:
            code_tokens += tree_to_token_index(child)
        return code_tokens


def tree_to_variable_index(root_node, index_to_code):
    if (len(root_node.children) == 0 or root_node.type == 'string' or root_node.type == 'comment' or 'comment' in root_node.type):
        index = (root_node.start_point, root_node.end_point)
        _, code = index_to_code[index]
        if root_node.type != code:
            return [(root_node.start_point, root_node.end_point)]
        else:
            return []
    else:
        code_tokens = []
        for child in root_node.children:
            code_tokens += tree_to_variable_index(child, index_to_code)
        return code_tokens


def index_to_code_token(index, code):
    start_point = index[0]
    end_point = index[1]
    if start_point[0] == end_point[0]:
        s = code[start_point[0]][start_point[1]:end_point[1]]
    else:
        s = ""
        s += code[start_point[0]][start_point[1]:]
        for i in range(start_point[0]+1, end_point[0]):
            s += " "+code[i]
        s += " "+code[end_point[0]][:end_point[1]]
    return s


parsers = {}
for lang in ['python', 'java']:
    try:
        LANGUAGE = Language('parser/my-languages.so', lang)
    except Exception as e:
        LANGUAGE = Language('../parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    parsers[lang] = parser


def travel(root_node, index_to_code, tokenizer):
    """Given a AST node, return AST travel sequence using Algo in the paper: https://arxiv.org/pdf/2203.03850.pdf"""
    if (len(root_node.children) == 0 or root_node.type == 'string' or root_node.type == 'comment' or 'comment' in root_node.type):
        index = (root_node.start_point, root_node.end_point)
        code = index_to_code[index][1]
        return tokenizer.tokenize(code)
    else:
        code_tokens = []
        for child in root_node.children:
            code_tokens += travel(child, index_to_code, tokenizer)
        # remove nodes that have only one children for reducing length
        if len(root_node.children) != 1:
            return ["AST#" + root_node.type.replace("#", "") + "#Left"] + code_tokens + ["AST#" + root_node.type.replace("#", "") + "#Right"]
        else:
            return code_tokens


def AST(code, lang, tokenizer):
    """Given a code, return its AST flatten sequence"""
    if lang == "php":
        code = "<?php "+code+"?>" 
    # remove comment
    try:
        code = remove_comments_and_docstrings(code, lang)
    except:
        pass
    # parse source code
    if lang == "csharp":
        tree = parsers["c_sharp"].parse(bytes(code, 'utf8'))
    else:
        tree = parsers[lang].parse(bytes(code, 'utf8'))  

    # obtain AST sequence
    root_node = tree.root_node  
    tokens_index = tree_to_token_index(root_node)
    code = code.split('\n')
    code_tokens = [index_to_code_token(x, code) for x in tokens_index]
    index_to_code = {}
    for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
        index_to_code[index] = (idx,code)  

    code_tokens = travel(root_node, index_to_code, tokenizer)
    return code_tokens


def get_fim_tokens(model_id):
    if 'qwen2.5' in model_id.lower():
        fim_tokens = FIMMAP['qwen2.5']
    elif 'starcoder' in model_id.lower():
        fim_tokens = FIMMAP['starcoder']
    else:
        raise NotImplementedError(f'No such model in FIM mapping: {model_id}')

    return fim_tokens


def truncate_trash(s, markers=["<|file_sep|><|fim_prefix|>", "<|fim_pad|>"]):
    indexes = [s.find(marker) for marker in markers]
    valid_indexes = [idx for idx in indexes if idx != -1]
    if valid_indexes:
        return s[:min(valid_indexes)]
    else:
        return s


class UniXcoderEncoder(nn.Module):
    """Wrapper Module for encoding inputs with UniXCoder model.
    Supports CLS pooling (default) and mean pooling.
    """
    def __init__(self, model, config, pooling="mean", normalize=True):
        super().__init__()
        self.model = model
        self.config = config
        assert pooling in ["cls", "mean"], "pooling must be 'cls' or 'mean'"
        self.pooling = pooling
        self.normalize = normalize

    @property
    def device(self):
        return self.model.device

    def forward(self, input_ids, pooling=None):
        if pooling is None:
            pooling = self.pooling

        attn_mask = input_ids.ne(self.config.pad_token_id)
        out = self.model(input_ids=input_ids, attention_mask=attn_mask)
        hidden = out.last_hidden_state  # [batch, seq_len, hidden_dim]

        if pooling == "cls":
            embeddings = hidden[:, 0, :]  # first token hidden space
        elif pooling == "mean":
            # mask out pad tokens before mean
            masked_hidden = hidden * attn_mask.unsqueeze(-1)
            sum_hidden = masked_hidden.sum(dim=1)
            lengths = attn_mask.sum(dim=1, keepdim=True)
            embeddings = sum_hidden / lengths.clamp(min=1e-9)
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return None, embeddings


class LlavaCodeConfig(PretrainedConfig):

    model_type = "llava_next"
    sub_configs = {"text_config": AutoConfig, "structure_config": AutoConfig}

    def __init__(
        self,
        structure_config=None,
        text_config=None,
        structure_token_id=None,
        pad_token_id=0,
        projector_hidden_act="gelu",
        tie_word_embeddings=False,
        multimodal_projector_bias=True,
        projector='3L',
        quantize=False,
        **kwargs,
    ):
        self.projector_hidden_act = projector_hidden_act
        self.multimodal_projector_bias = multimodal_projector_bias
        self.projector = projector

        self.structure_config = structure_config

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "llama"
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["llama"]()

        self.text_config = text_config

        if quantize:
            self.quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            self.quantization_config = {}

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

        self.structure_token_id = structure_token_id
        self.pad_token_id = pad_token_id  # has to go after super() init


class LlavaCodeMultiModalProjector3L(nn.Module):
    def __init__(self, config: LlavaCodeConfig):
        super().__init__()
        self.linear_1 = nn.Linear(
            config.structure_config.hidden_size,
            config.text_config.hidden_size * 2,
            bias=config.multimodal_projector_bias,
        )
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(
            config.text_config.hidden_size * 2, config.text_config.hidden_size * 2, bias=config.multimodal_projector_bias
        )
        self.linear_3 = nn.Linear(
            config.text_config.hidden_size * 2, config.text_config.hidden_size, bias=config.multimodal_projector_bias
        )
        self.ln_1 = nn.LayerNorm(config.text_config.hidden_size * 2)
        self.ln_2 = nn.LayerNorm(config.text_config.hidden_size * 2)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, structure_features):
        hidden_states = self.linear_1(structure_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.ln_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.linear_3(hidden_states)
        return hidden_states


class LlavaCodePreTrainedModel(PreTrainedModel):
    config_class = LlavaCodeConfig
    base_model_prefix = ""
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_cache_class = True
    _supports_flash_attn_2 = False
    _supports_sdpa = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_flex_attn = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        std = getattr(self.config, "initializer_range", self.config.get_text_config().initializer_range)

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, LlavaCodeModel):
            embed_std = 1 / math.sqrt(self.config.text_config.hidden_size)


@dataclass
class LlavaCodeCausalLMOutputWithPast(ModelOutput):
    """
    Base class for LlavaNext causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        structure_hidden_states (`torch.FloatTensor`, *optional*):
            A `torch.FloatTensor` of size (batch_size * num_patches, num_images, sequence_length, hidden_size)`.
            structure_hidden_states of the model produced by the structure encoder and after projecting the last hidden state.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    structure_features: Optional[torch.FloatTensor] = None
    structure_embeddings: Optional[torch.FloatTensor] = None


class LlavaCodeModel(LlavaCodePreTrainedModel):
    _checkpoint_conversion_mapping = {"language_model.model": "language_model"}

    def __init__(self, config: LlavaCodeConfig, model_path=None):
        super().__init__(config)

        if model_path:
            state_dict = torch.load(model_path, map_location="cuda")["state_dict"]
            state_dict = {
                k.removeprefix("model."): v
                for k, v in state_dict.items()
                if k.startswith("model.")}

            encoder = AutoModel.from_config(
                AutoConfig.from_pretrained(self.config.structure_config.model_id))
            self.language_model = AutoModelForCausalLM.from_config(
                AutoConfig.from_pretrained(self.config.text_config.model_id))
        else:
            encoder = AutoModel.from_pretrained(self.config.structure_config.model_id)
            self.language_model = AutoModelForCausalLM.from_pretrained(self.config.text_config.model_id)

        if 'unixcoder' in self.config.structure_config.model_id.lower():
            self.structure_model = UniXcoderEncoder(
                encoder, config=self.config.structure_config)
        elif 'qwen' in self.config.structure_config.model_id.lower():
            self.structure_model = QwenEmbedEncoder(
                encoder, config=self.config.structure_config)
        else:
            raise ValueError(f'Unrecognized structure model: {self.structure_model}')

        if self.config.projector == '3L':
            self.multi_modal_projector = LlavaCodeMultiModalProjector3L(config)
        elif self.config.projector == '4L':
            self.multi_modal_projector = LlavaCodeMultiModalProjector4L(config)
        else:
            raise ValueError(f'Unrecognized projector config: {self.config.projector }')

        self.vocab_size = config.text_config.vocab_size

        self.fim_tokens = get_fim_tokens(self.config.text_config.model_id)

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.post_init()

        if model_path:
            self.load_state_dict(state_dict)

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_structure_features_and_embeddings(self, structure_values, nums_structure_tokens=None, structure_pos_idx=None, structure_attn_mask=None):
        if structure_pos_idx and structure_attn_mask:

            # structure values: code + dfg traversal
            structure_pos_idx = structure_pos_idx.reshape(-1, 512)
            structure_attn_mask = structure_attn_mask.reshape(-1, 512, 512)

            nodes_mask = structure_pos_idx.eq(0)
            token_mask = structure_pos_idx.ge(2)

            inputs_embeddings = self.structure_model.roberta.embeddings.word_embeddings(structure_values.reshape(-1, 512))
            nodes_to_token_mask = nodes_mask[:, :, None] & token_mask[:, None, :] & structure_attn_mask
            nodes_to_token_mask = nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:, :, None]
            avg_embeddings = torch.einsum("abc,acd->abd", nodes_to_token_mask, inputs_embeddings)
            inputs_embeddings = inputs_embeddings*(~nodes_mask)[:, :, None] + avg_embeddings*nodes_mask[:, :, None]
            print('inputs_emb shape', inputs_embeddings.shape, 'attn mask:', structure_attn_mask.shape, 'pos_idx', structure_pos_idx.shape)

            outputs = self.structure_model.roberta(
                inputs_embeds=inputs_embeddings, attention_mask=structure_attn_mask,
                position_ids=structure_pos_idx, token_type_ids=structure_pos_idx.eq(-1).long())[0]
            structure_embedding = (outputs * token_mask.unsqueeze(-1)).sum(1) / token_mask.sum(-1).unsqueeze(-1)
        elif structure_pos_idx is None and structure_attn_mask is None:

            _, structure_embedding = self.structure_model(structure_values.reshape(-1, 512))  # unixcoder and jina take care of attention mask inside the forward method
        else:

            raise ValueError('Incorrect inputs to get_structure_features()')
        # print(f'Shape of Embedding output: {structure_embedding.shape}')

        if nums_structure_tokens is not None:
            # nums_structure_tokens: number of structure tokens for each sample in a batch
            max_num = nums_structure_tokens.max()
            row_ids = torch.arange(max_num).expand(len(nums_structure_tokens), max_num).to(nums_structure_tokens.device)
            mask = row_ids < nums_structure_tokens.unsqueeze(1)
            flat_mask = mask.flatten()
            # taking only those features that correspond to code_structure tokens
            # others fully consist of padding
            structure_embedding = structure_embedding[flat_mask]

        # structure_embedding = torch.nn.functional.normalize(structure_embedding, p=2, dim=-1)  # normalize the embedding
        # structure_embedding = torch.randn_like(structure_embedding, dtype=torch.float)  # sanity check with random inputs
        structure_features = self.multi_modal_projector(structure_embedding)
        # print(f'Shape of Projector output: {structure_features.shape}')
        return structure_features, structure_embedding

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        structure_values: torch.LongTensor = None,
        structure_features: torch.FloatTensor = None,
        structure_attn_mask: torch.Tensor = None,
        structure_pos_idx: torch.LongTensor = None,
        num_structure_tokens: torch.IntTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[LossKwargs]
    ) -> Union[Tuple, LlavaCodeCausalLMOutputWithPast]:
        r"""
        """
        # checking if cache is already in use (use_cache=True and iter > 1)
        # this is kostyl for starcoder
        if past_key_values is None:
            using_cache = False
        elif isinstance(past_key_values, list):
            using_cache = True
        elif isinstance(past_key_values, DynamicCache):
            using_cache = bool(past_key_values.key_cache)
        else:
            raise ValueError('Unknown past_key_values instance')

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)  # from language model only

        if structure_values is not None and structure_features is None and not using_cache:
            structure_features, structure_embeddings = self.get_structure_features_and_embeddings(
                structure_values, num_structure_tokens, structure_attn_mask=structure_attn_mask, structure_pos_idx=structure_pos_idx)
            structure_features = structure_features.to(inputs_embeds.device, inputs_embeds.dtype)
        else:
            structure_embeddings = None

        if structure_features is not None:
            special_structure_mask = (input_ids == self.config.structure_token_id).unsqueeze(-1)
            special_structure_mask = special_structure_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
            assert inputs_embeds[special_structure_mask].numel() == structure_features.numel(), \
                f'Mask does not correspond to the number of structure features: {inputs_embeds[special_structure_mask].numel()} != {structure_features.numel()}'
            inputs_embeds = inputs_embeds.masked_scatter(special_structure_mask, structure_features)

        outputs = self.language_model(
            # input_ids: Optional[torch.Tensor] = None,
            attention_mask=attention_mask,
            # token_type_ids: Optional[torch.Tensor] = None,
            position_ids=position_ids,
            past_key_values=past_key_values if using_cache else None,
            # head_mask: Optional[torch.Tensor] = None,
            inputs_embeds=inputs_embeds,
            # encoder_hidden_states: Optional[torch.Tensor] = None,
            # encoder_attention_mask: Optional[torch.Tensor] = None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs
        )

        return LlavaCodeCausalLMOutputWithPast(
            logits=outputs.logits,
            # last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            # hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            structure_features=structure_features,
            structure_embeddings=structure_embeddings
        )


class LlavaCodeForConditionalGeneration(LlavaCodePreTrainedModel, GenerationMixin):
    _checkpoint_conversion_mapping = {
        "^language_model.model": "model.language_model",
        "^structure_model": "model.structure_model",
        "^multi_modal_projector": "model.multi_modal_projector",
        # "^language_model.lm_head": "lm_head",
    }
    # _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: LlavaCodeConfig, model_path=None):
        super().__init__(config)
        self.model = LlavaCodeModel(config, model_path)
        # self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        self.vocab_size = self.config.text_config.vocab_size
        # self.language_model.resize_token_embeddings(self.vocab_size)
        self.pad_token_id = config.pad_token_id
        self.tokenizer = AutoTokenizer.from_pretrained(config.text_config.model_id, use_fast=False)
        self.tokenizer.add_tokens([STRUCTURE_TOKEN])
        if self.tokenizer.pad_token_id is None:  # case with starcoder
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        # print(f'Structure token: {STRUCTURE_TOKEN}')
        self.post_init()

    def set_trainer_args(self, trainer_args):
        self.trainer_args = trainer_args

    def setup(self, stage):
        # Loss Configuration
        if self.trainer_args.loss == 'mle':
            self.loss = nn.CrossEntropyLoss(ignore_index=-100)
        # elif self.trainer_args.loss == 'mse':
        #     self.loss = nn.MSELoss(reduction='mean')
        # elif self.trainer_args.loss == 'cosine':
        #     self.loss = lambda x, y: (1 - F.cosine_similarity(x, y)).mean()
        else:
            raise ValueError(f'Invalid loss: {self.trainer_args.loss}')

        self.alpha_kl = self.trainer_args.alpha_kl
        self.kl_temperature = self.trainer_args.kl_temperature
        self.distill_topk = self.trainer_args.distill_topk

        self.alpha_align = self.trainer_args.alpha_align
        self.alpha_scst = self.trainer_args.alpha_scst
        self.alpha_ce = self.trainer_args.alpha_ce

        self.reward = self.trainer_args.reward

        if stage == 'fit':
            # Hyperparameters and Configuration
            self.num_nodes = self.trainer_args.num_nodes
            self.dropout_p = self.trainer_args.dropout_p
            self.functional_dropout = self.trainer_args.functional_dropout

            self.lr = self.trainer_args.lr
            self.weight_decay = self.trainer_args.weight_decay
            self.num_warmup_steps = self.trainer_args.warmup_steps
            self.num_epochs = self.trainer_args.max_epochs
            self.train_batch_size = self.trainer_args.train_batch_size
            self.num_train_examples = self.trainer_args.num_training_examples
            self.num_gpu_per_node = self.trainer_args.devices
            self.accumulate_grad_batches = self.trainer_args.accumulate_grad_batches

            if self.trainer_args.max_steps == -1:
                num_steps_per_epoch = self.num_train_examples // (self.num_gpu_per_node * self.num_nodes * self.accumulate_grad_batches)
                self.num_training_steps = self.num_epochs * num_steps_per_epoch
                print(f"steps_per_epoch: {num_steps_per_epoch}\t total_training_steps: {self.num_training_steps}.")
            else:
                self.num_training_steps = self.trainer_args.max_steps

            self.lr_scheduler_type = self.trainer_args.lr_scheduler_type
            self.world_size = self.trainer_args.devices * self.num_nodes

            self.training_stage = self.trainer_args.training_stage

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @property
    def language_model(self):
        return self.model.language_model

    @property
    def lm_head(self):
        return self.model.language_model.lm_head

    @property
    def multi_modal_projector(self):
        return self.model.multi_modal_projector

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        structure_values: torch.LongTensor = None,
        structure_features: torch.FloatTensor = None,
        structure_attn_mask: torch.Tensor = None,
        structure_pos_idx: torch.LongTensor = None,
        num_structure_tokens: torch.IntTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[LossKwargs],
    ) -> Union[Tuple, LlavaCodeCausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.model(
            input_ids,
            structure_values=structure_values,
            structure_features=structure_features,
            structure_attn_mask=structure_attn_mask,
            structure_pos_idx=structure_pos_idx,
            num_structure_tokens=num_structure_tokens,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        return LlavaCodeCausalLMOutputWithPast(
            loss=None,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            # hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            structure_features=outputs.structure_features,
            structure_embeddings=outputs.structure_embeddings
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        attention_mask=None,
        cache_position=None,
        logits_to_keep=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward structure values to the model

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        return model_inputs

    def get_inputs_and_labels_fim(self, token_ids):
        """Prepares inputs and labels for Fill-in-the-Middle (FIM) training.

        Args:
            token_ids: Tensor of shape (batch, seq_len) with the rearranged FIM sequence.

        Returns:
            inp_tensor: input ids for model
            lbl_tensor: labels with -100 where no loss should be computed
            attention_mask: mask for padding tokens
        """
        # Standard causal LM shift
        inp_tensor = token_ids[:, :-1].clone()
        lbl_tensor = token_ids[:, 1:].clone()

        fim_middle_id = self.tokenizer.convert_tokens_to_ids(self.model.fim_tokens)[2]

        # Find first occurrence of <|fim_middle|> in each sequence
        fim_middle_mask = (lbl_tensor == fim_middle_id)
        # Convert boolean mask to index positions
        # argmax works because <|fim_middle|> appears exactly once
        middle_pos = fim_middle_mask.float().argmax(dim=1)

        # Build a position index tensor for broadcasting
        seq_len = lbl_tensor.size(1)
        pos_ids = torch.arange(seq_len, device=lbl_tensor.device).unsqueeze(0)  # [1, seq_len]

        # Mask: keep tokens where position > middle_pos
        keep_mask = pos_ids > middle_pos.unsqueeze(1)

        # Apply mask and pad masking
        lbl_tensor = torch.where(keep_mask, lbl_tensor, torch.full_like(lbl_tensor, -100))
        lbl_tensor[lbl_tensor == self.pad_token_id] = -100

        attention_mask = inp_tensor.ne(self.pad_token_id)

        return inp_tensor, lbl_tensor, attention_mask

    def similarity_measure(self, pred, gold, strip=False):
        if max(len(pred), len(gold)) == 0:
            return 1.0, 1.0, 1.0, 1.0  # both empty → perfect match

        gold_text = self.tokenizer.decode(gold, skip_special_tokens=True)
        num_lines = len(gold_text.split('\n'))

        # ids out of tokenizer vocab might occur
        pred_tokens = self.tokenizer.convert_ids_to_tokens(pred, skip_special_tokens=True)
        pred_tokens = [t for t in pred_tokens if t is not None] 
        pred_text = self.tokenizer.convert_tokens_to_string(pred_tokens)
        pred_text = truncate_trash(pred_text)  # truncating everyting after the first transh token

        pred_text = "\n".join(pred_text.split('\n')[:num_lines])

        if strip:  # only strip during validation
            pred_text = pred_text.strip()
            gold_text = gold_text.strip()

        # cumulative precision 1 / L \sum_{i=1}^L P@i
        def cumulative_precision(s_gold, s_pred):
            cum_prec = 0.0
            hits = 0
            n = len(s_gold)
            for i in range(n):
                if i < len(s_pred) and s_gold[i] == s_pred[i]:
                    hits += 1
                cum_prec += hits / (i + 1)
            cum_prec = cum_prec / n
            return cum_prec

        def weighted_jaccard(s_gold, s_pred):
            n = len(s_gold)
            if n == 0:
                return 1.0

            weights = 1.0 / torch.arange(1, n + 1, dtype=torch.float32)
            match_mask = torch.zeros(n, dtype=torch.float32)
            for i in range(n):
                if i < len(s_pred) and s_gold[i] == s_pred[i]:
                    match_mask[i] = 1.0

            intersection = (weights * match_mask).sum()
            union = weights.sum()

            return (intersection / union).item()

        cum_prec = cumulative_precision(gold_text, pred_text)
        wji = weighted_jaccard(gold_text, pred_text)

        # for debugging
        # import json
        # record = {'pred': pred_text, 'target': gold_text}
        # with open('ast_cfc_java.jsonl', 'a') as f:
        #     f.write(json.dumps(record) + "\n")

        es = 1 - editdistance.eval(pred_text, gold_text) / max(len(pred_text), len(gold_text))

        def tokenize_code(code):
            code = re.sub(r"([^A-Za-z0-9_])", r" \1 ", code)
            code = re.sub(r"([a-z])([A-Z])", r"\1 \2", code)
            code = re.sub(r"\s+", " ", code)
            code = code.replace('"', "`")
            code = code.replace("'", "`")
            tokens = [t for t in code.split(" ") if t]
            return tokens

        em = (tokenize_code(pred_text) == tokenize_code(gold_text))

        return em, es, cum_prec, wji

    def validation_step(self, batch, batch_idx):

        token_ids, structure_ids = batch['input_ids'], batch['structure_ids']
        # token_ids, structure_ids = batch['teacher_input_ids'], batch['structure_ids']
        num_structure_tokens, structure_attn_mask, structure_pos_idx = batch.get('num_structure_tokens'), batch.get('structure_attn_mask'), batch.get('structure_pos_idx')
        input_ids, labels, attention_mask = self.get_inputs_and_labels_fim(token_ids)

        with torch.no_grad():

            align_loss = torch.tensor(0.0, device=input_ids.device)
            var_loss = torch.tensor(0.0, device=input_ids.device)
            ce_loss = torch.tensor(0.0, device=input_ids.device)
            scst_loss = torch.tensor(0.0, device=input_ids.device)
            em = torch.tensor(0.0, device=input_ids.device)
            es = torch.tensor(0.0, device=input_ids.device)
            cum_prec = torch.tensor(0.0, device=input_ids.device)
            wji = torch.tensor(0.0, device=input_ids.device)
            loss = torch.tensor(0.0, device=input_ids.device)

            outputs = self(
                input_ids=input_ids,
                attention_mask=attention_mask,
                structure_values=structure_ids,
                structure_attn_mask=structure_attn_mask,
                structure_pos_idx=structure_pos_idx,
                num_structure_tokens=num_structure_tokens
            )
            logits = outputs.logits

            ce_loss = self.loss(logits.view(-1, self.vocab_size), labels.view(-1))
            loss += self.alpha_ce * ce_loss

            if self.alpha_align is not None and self.alpha_align > 0.0:

                projections = outputs.structure_features
                embeddings = outputs.structure_embeddings

                proj_sim = F.cosine_similarity(projections.unsqueeze(1), projections.unsqueeze(0), dim=-1)
                embed_sim = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)
                align_loss = F.mse_loss(proj_sim, embed_sim)

                loss += self.alpha_align * align_loss

            if self.alpha_scst is not None and self.alpha_scst > .0:
                assert input_ids.shape[0] == 1, 'Change the logic below'
                prompt_len = input_ids[labels == -100].shape[0] + 1
                # ===== Baseline: greedy decode =====
                greedy_ids = self.generate(
                    input_ids[:, :prompt_len],
                    attention_mask=attention_mask[:, :prompt_len],
                    structure_values=structure_ids,
                    num_structure_tokens=num_structure_tokens,
                    max_new_tokens=50, do_sample=False,
                    # pad_token_id=self.tokenizer.eos_token_id
                )
                em, es, cum_prec, wji = self.similarity_measure(greedy_ids[0, prompt_len:], labels[labels != -100], strip=True)

                # ---- Log probs of greedy tokens ----
                greedy_input_ids = greedy_ids[:, :-1]
                greedy_attention_mask = (greedy_input_ids != self.tokenizer.eos_token_id).long()
                greedy_logits = self(
                    input_ids=greedy_input_ids,
                    attention_mask=greedy_attention_mask,
                    structure_values=structure_ids,
                    num_structure_tokens=num_structure_tokens
                ).logits
                log_probs = F.log_softmax(greedy_logits, dim=-1)

                gen_tokens = greedy_ids[:, prompt_len:]
                gen_logits = log_probs[:, prompt_len-1:, :]
                seq_log_probs = gen_logits.gather(2, gen_tokens.unsqueeze(-1)).squeeze(-1)
                seq_log_prob = seq_log_probs.sum(dim=1)

                # ---- Greedy-imitation loss ----
                reward = eval(self.reward)
                scst_loss = -(reward * seq_log_prob).mean()

                loss += self.alpha_scst * scst_loss

        self.log("Val/Loss/MLE", ce_loss, sync_dist=True, on_epoch=True, prog_bar=True)
        self.log("Val/Loss/Align", align_loss, sync_dist=True, on_epoch=True, prog_bar=True)
        self.log("Val/Loss/Var", var_loss, sync_dist=True, on_epoch=True, prog_bar=True)
        self.log("Val_Acc_EM", em, sync_dist=True, on_epoch=True, prog_bar=True)
        self.log("Val_Acc_ES", es, sync_dist=True, on_epoch=True, prog_bar=True)
        self.log("Val_Acc_Precision", cum_prec, sync_dist=True, on_epoch=True, prog_bar=True)
        self.log("Val_Acc_WJI", wji, sync_dist=True, on_epoch=True, prog_bar=True)
        self.log("Val/Loss/SCST", scst_loss, sync_dist=True, on_epoch=True, prog_bar=True)
        self.log("Val/Loss/All", loss, sync_dist=True, on_epoch=True, prog_bar=True)

        return {"val_ce": ce_loss, 
                "val_align": align_loss, 
                "val_var": var_loss, 
                "val_em": em,
                "val_es": es, 
                "val_precision": cum_prec,
                "val_wji": wji,
                "val_scst": scst_loss, 
                "val_all": loss}

    def on_validation_epoch_end(self):

        # cleaning up memory
        torch.cuda.empty_cache()
        gc.collect()


def cal_edit_sim_repoeval(references, hypotheses):
    total = len(references)
    edit_sim = 0.0
    for pred, gt in zip(hypotheses, references):
        pred = pred.strip()
        gt = gt.strip()
        if max(len(pred), len(gt)) == 0:
            continue
        edit_sim += 1 - editdistance.eval(pred, gt) / max(len(pred), len(gt))
    return edit_sim / total


def tokenize_code(code):
    code = re.sub(r"([^A-Za-z0-9_])", r" \1 ", code)
    code = re.sub(r"([a-z])([A-Z])", r"\1 \2", code)
    code = re.sub(r"\s+", " ", code)
    code = code.replace('"', "`")
    code = code.replace("'", "`")
    tokens = [t for t in code.split(" ") if t]
    return tokens


def cal_exact_match(references, hypotheses):
    em_score = []
    for pred, gold in zip(hypotheses, references):
        em_score.append(tokenize_code(pred) == tokenize_code(gold))
    return np.mean(em_score)


def get_ast(parser, code):
    assert isinstance(code, str) or isinstance(code, bytes)
    if isinstance(code, str):
        code = bytes(code, "utf8")
    try:
        tree = parser.parse(code)
        return tree
    except Exception as e:
        return None


def is_parse_valid(parser, code):
    def syntax_error(node):
        if node.type == "ERROR":
            return True
        try:
            for child in node.children:
                if syntax_error(child):
                    return True
        except RecursionError as err:
            return True

        return False

    tree = get_ast(parser, code)
    if tree is not None:
        return not syntax_error(tree.root_node)
    return False


def get_valid_completion(prompt, completion, parser):
    for i in range(len(completion), -1, -1):
        code = prompt + completion[:i]
        if is_parse_valid(parser, code):
            return "parseable", completion[:i].rstrip()

    return "not_parseable", completion


def process_examples(task, args):
    sample, ex = args
    global parser

    prediction = sample["pred"]
    target = ex["groundtruth"]

    num_target_lines = sum([1 for l in target.split("\n") if l.strip()])
    pred_lines = [l for l in prediction.split("\n") if l.strip()][:num_target_lines]
    prediction = "\n".join(pred_lines)

    trunc_s = {
        "task_id": sample["task_id"],
        "pred": prediction,
        "target": target
    }
    return trunc_s


def compute_metric_stmt(args):
    with open(f"{args.output_dir}/prediction.jsonl", "r") as f_pred:
        samples = []
        for l in f_pred.readlines():
            samples.append(json.loads(l))

    examples = {}
    with open(args.prompt_file, "r") as f_in:
        for l in f_in.readlines():
            ex = json.loads(l)
            if hasattr(args, "focused_repo") and args.focused_repo and args.focused_repo not in re.sub('/', '_', ex['metadata']['repository']):
                continue
            examples[ex["metadata"]["task_id"]] = {
                "task_id": ex["metadata"]["task_id"],
                "prompt": ex["prompt"],
                "groundtruth": ex["groundtruth"]
            }

    assert len(samples) == len(examples), f"{len(samples)} != {len(examples)}"

    global parser
    # language = Language(args.ts_lib, "python")
    ts_lang = args.language
    if ts_lang == 'csharp':
        ts_lang = 'c_sharp'
    language = Language('parser/my-languages.so', ts_lang)
    parser = Parser()
    parser.set_language(language)

    truncated_samples = []
    print("post-processing samples ...")
    pool = mp.Pool(mp.cpu_count() - 1)
    worker = partial(process_examples, args.task)

    with tqdm(total=len(samples)) as pbar:
        for trunc_s in pool.imap_unordered(worker, zip(samples, [examples[s["task_id"]] for s in samples])):
            truncated_samples.append(trunc_s)
            pbar.update()

    with open(f"{args.output_dir}/prediction_truncated.jsonl", 'w', encoding="utf-8") as pt:
        for trunc_s in truncated_samples:
            pt.write(json.dumps(trunc_s) + "\n")

    # Score calculation

    detailed_results = []
    exact_match = 0
    # edit_sim = 0
    edit_sim_repoeval = 0

    for idx, trunc_s in enumerate(truncated_samples):
        # es = cal_edit_sim([trunc_s["target"]], [trunc_s["pred"]])
        es_repoeval = cal_edit_sim_repoeval([trunc_s["target"]], [trunc_s["pred"]])
        em = cal_exact_match([trunc_s["target"]], [trunc_s["pred"]])
        # edit_sim += es
        edit_sim_repoeval += es_repoeval
        exact_match += em

        detailed_results.append({
            "task_id": trunc_s["task_id"],
            "em": em,
            # "es": es,
            "es_repoeval": es_repoeval
        })

    em_ratio = round(exact_match / len(truncated_samples) * 100, 2)
    # edit_sim = round(edit_sim / len(truncated_samples), 2)
    edit_sim_repoeval = round(edit_sim_repoeval / len(truncated_samples) * 100, 2)

    print(
        f"Code Matching: "
        f"EM {em_ratio:.2f}, "
        # f"ES {edit_sim:.2f}, "
        f"ES {edit_sim_repoeval:.2f}"
    )

    with open(f"{args.output_dir}/detailed_results.json", 'w') as f:
        for dr in detailed_results:
            f.write(json.dumps(dr) + "\n")

    # write the results to a file
    with open(f"{args.output_dir}/results.json", 'w') as f:
        res = {
            "em": em_ratio,
            # "es": edit_sim,
            "es": edit_sim_repoeval,
            "total": len(truncated_samples)
        }
        f.write(json.dumps(res, indent=2))


def prepare_prompt(args,
                   tokenizer,
                   structure_tokenizer,
                   fim_tokens,
                   entry):
    """Dataset type: 10 chunks of cross-file context, 10 lines each, stored as an array
    """
    fim_prefix, fim_suffix, fim_middle = fim_tokens

    left_cxt = entry["prompt"]
    right_cxt = entry["right_context"]
    if 'crossfile_context' in entry:
        crossfile_cxt = entry["crossfile_context"] if type(entry["crossfile_context"]) == str else entry["crossfile_context"]['text']
    else:
        crossfile_cxt = None

    if args.data_prefix == 'ast_cfc':
        # splitting context into chunks
        # one chunk for one file
        lines = crossfile_cxt.splitlines()[1:]  # removing the "Here are some examples..." line
        skip = False
        current_lines = []
        chunks = []
        for line in lines:
            if 'the below code fragment can be found in:' in line:
                skip = True
                if current_lines:
                    chunks.append('\n'.join(current_lines))
                current_lines = []
            elif skip:  # skipping the file path
                skip = False
            elif line:
                current_lines.append(line.strip())
        if current_lines:
            chunks.append('\n'.join(current_lines))

        # restrict number of injection tokens (RAG files)
        num_structure_tokens = min(args.num_structure_tokens, len(chunks))
        chunks = chunks[:num_structure_tokens]

        structure_ids = torch.empty(0, dtype=torch.long)
        for cfc in chunks:
            ast_tokens = AST(cfc.replace('#', ''), args.language, structure_tokenizer)  # decommenting
            ast_tokens = ast_tokens[:args.max_structure_length - 4]  # 4 special tokens for unixcoder
            chunk_tokens = [structure_tokenizer.cls_token, "<encoder-only>", structure_tokenizer.sep_token] \
                + ast_tokens + [structure_tokenizer.sep_token]
            chunk_ids = structure_tokenizer.convert_tokens_to_ids(chunk_tokens)
            structure_ids = torch.hstack([structure_ids, F.pad(torch.tensor(chunk_ids), (0, args.max_structure_length-len(chunk_ids)), value=structure_tokenizer.pad_token_id)])

        left_cxt_truncated = tokenizer.decode(tokenizer.encode(left_cxt)[-(args.max_seq_length - args.right_context_length - 3):])
        right_cxt_truncated = tokenizer.decode(tokenizer.encode(right_cxt)[:args.right_context_length])

        if args.cfc_place == 'premiddle':
            # prompt = f"{fim_prefix}{left_cxt_truncated}{fim_suffix}{right_cxt_truncated}\n# Here are some relevant code fragments from other files of the repo:{STRUCTURE_TOKEN * num_structure_tokens}{fim_middle}"
            prompt = f"{fim_prefix}{left_cxt_truncated}{fim_suffix}{right_cxt_truncated}{STRUCTURE_TOKEN * num_structure_tokens}{fim_middle}"
        elif args.cfc_place == 'preprefix':
            prompt = f"{STRUCTURE_TOKEN * num_structure_tokens}{fim_prefix}{left_cxt_truncated}{fim_suffix}{right_cxt_truncated}{fim_middle}"

        return prompt, structure_ids, torch.tensor([num_structure_tokens])

    elif args.data_prefix == 'default':

        lr_budget = args.max_seq_length - args.gen_length - 3  # 3 tokens for FIM
        rc_budget = int(lr_budget / (args.lc_rc_ratio + 1))
        lc_budget = int(rc_budget * args.lc_rc_ratio)

        left_cxt_truncated = tokenizer.decode(tokenizer.encode(left_cxt)[-lc_budget:])
        right_cxt_truncated = tokenizer.decode(tokenizer.encode(right_cxt)[:rc_budget])

        prompt = f"{fim_prefix}{left_cxt_truncated}{fim_suffix}{right_cxt_truncated}{fim_middle}"

        return prompt, None, None

    elif args.data_prefix == 'default_cfc':

        assert crossfile_cxt is not None

        # making the same lr_budget as for other experiments, not considering cfc length
        lr_budget = args.max_seq_length - args.gen_length - 3  # 3 tokens for FIM
        rc_budget = int(lr_budget / (args.lc_rc_ratio + 1))
        lc_budget = int(rc_budget * args.lc_rc_ratio)

        left_cxt_truncated = tokenizer.decode(tokenizer.encode(left_cxt)[-lc_budget:])
        right_cxt_truncated = tokenizer.decode(tokenizer.encode(right_cxt)[:rc_budget])
        crossfile_cxt_truncated = tokenizer.decode(tokenizer.encode(crossfile_cxt)[:args.cfc_seq_length])

        if args.cfc_place == 'premiddle':
            prompt = f'{fim_prefix}{left_cxt_truncated}{fim_suffix}{right_cxt_truncated}{crossfile_cxt_truncated}{fim_middle}'
        elif args.cfc_place == 'preprefix':
            prompt = f'{crossfile_cxt_truncated}{fim_prefix}{left_cxt_truncated}{fim_suffix}{right_cxt_truncated}{fim_middle}'

        return prompt, None, None

    else:
        raise NotImplementedError


def build_dataset(args, code_tokenizer, ast_tokenizer, fim_tokens):
    with open(args.prompt_file) as f:
        raw_data = [json.loads(line) for line in f.readlines()]

    data = []
    for entry in raw_data:

        entry['llm_prompt'], entry['structure_ids'], entry['num_structure_tokens'] = \
            prepare_prompt(args, code_tokenizer, ast_tokenizer, fim_tokens, entry)
        data.append(entry)

    return data


if __name__ == "__main__":

    args = ArgsMock()

    code_tokenizer = AutoTokenizer.from_pretrained(args.text_model_id, use_fast=False)
    code_tokenizer.add_tokens([STRUCTURE_TOKEN])
    if code_tokenizer.pad_token_id is None:
        code_tokenizer.pad_token_id = code_tokenizer.eos_token_id
    structure_token_id = code_tokenizer.convert_tokens_to_ids(STRUCTURE_TOKEN)

    structure_tokenizer = AutoTokenizer.from_pretrained(args.structure_model_id, use_fast=False)

    structure_config = AutoConfig.from_pretrained(args.structure_model_id)
    structure_config.model_id = args.structure_model_id
    structure_config.pad_token_id = structure_tokenizer.pad_token_id

    text_config = AutoConfig.from_pretrained(args.text_model_id)
    text_config.model_id = args.text_model_id
    assert len(code_tokenizer) <= text_config.vocab_size, 'The tokenizer length is larger than the embedding layer shape, resize the embeddings'
    configuration = LlavaCodeConfig(structure_config, text_config,
                                    pad_token_id=code_tokenizer.pad_token_id,
                                    structure_token_id=structure_token_id,
                                    projector=args.projector)

    if args.model_checkpoint is not None:
        print(f"Loading checkpoint: {args.model_checkpoint}")
        model = LlavaCodeForConditionalGeneration(
            config=configuration, model_path=args.model_checkpoint)
    else:
        model = LlavaCodeForConditionalGeneration(configuration)
    if args.projector_checkpoint:
        print(f'Loading projection weighs from {args.projector_checkpoint}')
        model.multi_modal_projector.load_state_dict(torch.load(args.projector_checkpoint))
    model = model.to(device)
    model.eval()

    fim_tokens = get_fim_tokens(args.text_model_id)
    # print('fim tokens:', fim_tokens)
    data = build_dataset(args, code_tokenizer, structure_tokenizer, fim_tokens)
    # print(f'Number of samples: {len(data)}')

    all_preds = []
    for entry in tqdm(data):
        with torch.no_grad():
            inputs = code_tokenizer(entry['llm_prompt'], return_tensors='pt').to(device)
            cut_at = inputs.input_ids.shape[1]

            num_structure_tokens = entry['num_structure_tokens']
            print(f'Shape of the Input Sequence: {inputs.input_ids.shape} with {num_structure_tokens} context tokens')
            if not args.data_prefix.startswith('default') and num_structure_tokens > 0:
                structure_ids = entry['structure_ids'].to(device)
                cur_pred = model.generate(  # Use model.module to access the original model inside DDP
                    **inputs,
                    do_sample=args.do_sample,
                    structure_values=structure_ids,
                    num_structure_tokens=num_structure_tokens.to(device),
                    max_new_tokens=args.gen_length)
            else:
                cur_pred = model.generate(
                    **inputs,
                    do_sample=args.do_sample,
                    max_new_tokens=args.gen_length)

            prediction = code_tokenizer.decode(cur_pred[0][cut_at:], skip_special_tokens=True)
            prediction = truncate_trash(prediction)
            print(f"Shape of the Output sequence: {code_tokenizer(prediction, return_tensors='pt').input_ids.shape}")

            all_preds.append({
                "task_id": entry["metadata"]["task_id"],
                "pred": prediction,
            })
            print(f'Prediction:\n{prediction}')

    os.makedirs(args.output_dir, exist_ok=True)
    with open(f"{args.output_dir}/prediction.jsonl", "w", encoding="utf-8") as f_pred:
        for entry in all_preds:
            if isinstance(entry, list):
                for entry_ in entry:
                    f_pred.write(json.dumps(entry_) + "\n")
            else:
                f_pred.write(json.dumps(entry) + "\n")

    compute_metric_stmt(args)
