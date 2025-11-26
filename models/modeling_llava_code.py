from transformers import (
    PreTrainedModel,
    RobertaForSequenceClassification,
    AutoModel,
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
    PretrainedConfig,
    GenerationMixin,
)

from transformers.utils import can_return_tuple, LossKwargs
from transformers.processing_utils import Unpack
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers.trainer_pt_utils import get_parameter_names
from transformers.optimization import (
    get_linear_schedule_with_warmup,
    get_inverse_sqrt_schedule,
    get_cosine_schedule_with_warmup
)
from transformers.cache_utils import DynamicCache

from lightning.pytorch import LightningModule

import torch
import torch.nn as nn
from torch.optim import AdamW
import torch.nn.functional as F
import editdistance
import warnings
import math
import gc
import re

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from .config import LlavaCodeConfig
from .projector import LlavaCodeMultiModalProjector3L, LlavaCodeMultiModalProjector4L
from .modeling_unixcoder import UniXcoderEncoder
from .modeling_gnn_encoder import EnhancedGNNEncoder
from .modeling_jina import JinaEncoder
from .modeling_qwenembed import QwenEmbedEncoder
from .kl_loss import get_kl_loss

from datamodule.const import STRUCTURE_TOKEN, FIMMAP
from datamodule.utils import get_fim_tokens, truncate_trash


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

    def __init__(self, config: LlavaCodeConfig):
        super().__init__(config)
        if 'unixcoder' in self.config.structure_config.model_id.lower():
            # self.structure_model = UniXcoder(self.config.structure_config.model_id)
            self.structure_model = UniXcoderEncoder(
                AutoModel.from_pretrained(self.config.structure_config.model_id), config=self.config.structure_config)
        elif 'qwen' in self.config.structure_config.model_id.lower():
            self.structure_model = QwenEmbedEncoder(
                AutoModel.from_pretrained(self.config.structure_config.model_id), config=self.config.structure_config)
        # elif 'gnn_encoder' in self.config.structure_config.model_id.lower():
        #     self.structure_model = EnhancedGNNEncoder(
        #         hidden_size=self.config.structure_config.hidden_size, num_node_types=self.config.structure_config.num_node_types)
        #     self.structure_model.load_state_dict(torch.load(self.config.structure_config.model_id))
        elif 'graphcodebert' in self.config.structure_config.model_id.lower():
            self.structure_model = RobertaForSequenceClassification.from_pretrained(self.config.structure_config.model_id, config=self.config.structure_config)
        elif 'jina' in self.config.structure_config.model_id.lower():
            self.structure_model = JinaEncoder(
                model=AutoModel.from_pretrained(self.config.structure_config.model_id, trust_remote_code=True), config=self.config.structure_config)
        else:
            raise ValueError(f'Unrecognized structure model: {self.structure_model}')

        if self.config.projector == '3L':
            self.multi_modal_projector = LlavaCodeMultiModalProjector3L(config)
        elif self.config.projector == '4L':
            self.multi_modal_projector = LlavaCodeMultiModalProjector4L(config)
        else:
            raise ValueError(f'Unrecognized projector config: {self.config.projector }')
        # print('before post_init', self.multi_modal_projector.linear_1.weight.data.norm(2))

        self.vocab_size = config.text_config.vocab_size

        if self.config.quantization_config:
            self.language_model = AutoModelForCausalLM.from_pretrained(
                self.config.text_config.model_id,
                quantization_config=self.config.quantization_config,
                device_map=None,
                low_cpu_mem_usage=True
            )
        else:
            self.language_model = AutoModelForCausalLM.from_pretrained(self.config.text_config.model_id)

        self.fim_tokens = get_fim_tokens(self.config.text_config.model_id)

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.post_init()
        # print('post init', self.multi_modal_projector.linear_1.weight.data.norm(2))

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

        if self.injector is not None:
            self.injector.injection_tensor = torch.zeros_like(inputs_embeds)
        if structure_features is not None:
            special_structure_mask = (input_ids == self.config.structure_token_id).unsqueeze(-1)
            special_structure_mask = special_structure_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
            assert inputs_embeds[special_structure_mask].numel() == structure_features.numel(), \
                f'Mask does not correspond to the number of structure features: {inputs_embeds[special_structure_mask].numel()} != {structure_features.numel()}'
            if self.injector is not None:
                self.injector.injection_tensor = torch.zeros_like(inputs_embeds).to(inputs_embeds.device).masked_scatter(special_structure_mask, structure_features)
            else:
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


class LlavaCodeForConditionalGeneration(LlavaCodePreTrainedModel, GenerationMixin, LightningModule):
    _checkpoint_conversion_mapping = {
        "^language_model.model": "model.language_model",
        "^structure_model": "model.structure_model",
        "^multi_modal_projector": "model.multi_modal_projector",
        # "^language_model.lm_head": "lm_head",
    }
    # _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: LlavaCodeConfig):
        super().__init__(config)
        self.model = LlavaCodeModel(config)
        # self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        self.vocab_size = self.config.text_config.vocab_size
        # self.language_model.resize_token_embeddings(self.vocab_size)
        self.pad_token_id = config.pad_token_id
        self.tokenizer = AutoTokenizer.from_pretrained(config.text_config.model_id, use_fast=False)
        self.tokenizer.add_tokens([STRUCTURE_TOKEN])
        if self.tokenizer.pad_token_id is None:  # case with starcoder
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        print(f'Structure token: {STRUCTURE_TOKEN}')
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

    # def get_inputs_and_labels(self, token_ids, mask_prob=None):
    #     inp_tensor = token_ids[:, :-1].clone()

    #     lbl_tensor = token_ids[:, 1:].clone()
    #     lbl_tensor[lbl_tensor == self.pad_token_id] = -100

    #     attention_mask = inp_tensor.ne(self.pad_token_id)

    #     if mask_prob is not None:

    #         batch_size, seq_len = inp_tensor.shape

    #         start_pos = (inp_tensor == 151659).float().argmax(dim=1)
    #         end_pos = (inp_tensor == 151660).float().argmax(dim=1)

    #         position_ids = torch.arange(seq_len, device=inp_tensor.device).unsqueeze(0)
    #         in_mask_region = (position_ids > start_pos.unsqueeze(1)) & (position_ids < end_pos.unsqueeze(1))
    #         random_mask = torch.bernoulli(torch.full_like(inp_tensor, mask_prob, dtype=torch.float)).bool()
    #         final_mask = in_mask_region & random_mask

    #         # Loss masking
    #         lbl_tensor = torch.where(final_mask, torch.full_like(lbl_tensor, -100), lbl_tensor)

    #         # Attention mask — exclude pads AND masked region
    #         attention_mask = (~final_mask) & inp_tensor.ne(self.pad_token_id)

    #     return inp_tensor, lbl_tensor, attention_mask

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

    def training_step(self, batch, batch_idx):
        token_ids, structure_ids = batch['input_ids'], batch['structure_ids']
        # print('structure_values shape', structure_ids.shape,  'structure_values type', structure_ids.dtype)
        num_structure_tokens, structure_attn_mask, structure_pos_idx = batch.get('num_structure_tokens'), batch.get('structure_attn_mask'), batch.get('structure_pos_idx')

        input_ids, labels, attention_mask = self.get_inputs_and_labels_fim(token_ids)

        align_loss = torch.tensor(0.0, device=input_ids.device)
        var_loss = torch.tensor(0.0, device=input_ids.device)
        kl_loss = torch.tensor(0.0, device=input_ids.device)
        ce_loss = torch.tensor(0.0, device=input_ids.device)
        scst_loss = torch.tensor(0.0, device=input_ids.device)
        em = torch.tensor(0.0, device=input_ids.device)
        es = torch.tensor(0.0, device=input_ids.device)
        cum_prec = torch.tensor(0.0, device=input_ids.device)
        wji = torch.tensor(0.0, device=input_ids.device)
        loss = torch.tensor(0.0, device=input_ids.device)

        assert structure_attn_mask is None and structure_pos_idx is None
        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            structure_values=structure_ids,
            structure_attn_mask=structure_attn_mask,
            structure_pos_idx=structure_pos_idx,
            num_structure_tokens=num_structure_tokens)
        logits = outputs.logits
        ce_loss = self.loss(logits.view(-1, self.vocab_size), labels.view(-1))

        loss += self.alpha_ce * ce_loss

        if self.alpha_align is not None and self.alpha_align > .0:
            projections = outputs.structure_features
            embeddings = outputs.structure_embeddings

            proj_sim = F.cosine_similarity(projections.unsqueeze(1), projections.unsqueeze(0), dim=-1)
            embed_sim = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)

            align_loss = F.mse_loss(proj_sim, embed_sim)

            # proj_var = projections.var(dim=0).mean()
            # var_loss = F.relu(1e-4 - proj_var)

            # loss += self.alpha_align * align_loss + self.alpha_align / 10 * var_loss
            loss += self.alpha_align * align_loss

        if self.alpha_kl is not None and self.alpha_kl > .0:
            assert self.training_stage > 0
            with torch.no_grad():
                self.eval()
                teacher_input_ids, teacher_labels, teacher_attention_mask = self.get_inputs_and_labels_fim(batch['teacher_input_ids'])
                teacher_logits = self(
                    input_ids=teacher_input_ids,
                    attention_mask=teacher_attention_mask).logits
                self.train()

            kl_loss = get_kl_loss(
                teacher_logits=teacher_logits,
                teacher_labels=teacher_labels,
                student_logits=logits,
                student_labels=labels,
                temperature=self.kl_temperature,
            )

            loss += self.alpha_kl * kl_loss

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
                pad_token_id=self.tokenizer.eos_token_id
            )
            em, es, cum_prec, wji = self.similarity_measure(greedy_ids[0, prompt_len:], labels[labels != -100])

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

        # if self.alpha_scst is not None and self.alpha_scst > .0:
        #     assert input_ids.shape[0] == 1, 'Change the logic below'
        #     prompt_len = input_ids[labels == -100].shape[0] + 1
        #     # ===== Baseline: greedy decode =====
        #     greedy_ids = self.generate(
        #         input_ids[:, :prompt_len],
        #         attention_mask=attention_mask[:, :prompt_len],
        #         structure_values=structure_ids,
        #         num_structure_tokens=num_structure_tokens,
        #         max_new_tokens=50, do_sample=False,
        #         pad_token_id=self.tokenizer.eos_token_id
        #     )
        #     baseline_rewards = self.cal_exact_match(greedy_ids[0, prompt_len:], labels[labels != -100])

        #     # ===== Sampled decode (exploration) =====
        #     sampled_ids = self.generate(
        #         input_ids[:, :prompt_len],
        #         attention_mask=attention_mask[:, :prompt_len],
        #         structure_values=structure_ids,
        #         num_structure_tokens=num_structure_tokens,
        #         max_new_tokens=50, do_sample=True, top_p=0.9, temperature=1.0,
        #         pad_token_id=self.tokenizer.eos_token_id
        #     )
        #     sampled_rewards = self.cal_exact_match(sampled_ids[0, prompt_len:], labels[labels != -100])

        #     # ---- Log probs of sampled tokens ----
        #     sampled_input_ids = sampled_ids[:, :-1]
        #     sampled_attention_mask = (sampled_input_ids != self.pad_token_id).long()
        #     sampled_logits = self(
        #         input_ids=sampled_input_ids,
        #         attention_mask=sampled_attention_mask,
        #         structure_values=structure_ids,
        #         num_structure_tokens=num_structure_tokens).logits
        #     log_probs = F.log_softmax(sampled_logits, dim=-1)

        #     gen_tokens = sampled_ids[:, prompt_len:]
        #     gen_logits = log_probs[:, prompt_len-1:, :]
        #     seq_log_probs = gen_logits.gather(2, gen_tokens.unsqueeze(-1)).squeeze(-1)
        #     seq_log_prob = seq_log_probs.sum(dim=1)

        #     # ---- SCST loss ----
        #     rewards = torch.tensor(sampled_rewards, device=seq_log_prob.device, dtype=torch.float)
        #     baselines = torch.tensor(baseline_rewards, device=seq_log_prob.device, dtype=torch.float)
        #     advantages = rewards - baselines  # [B]
        #     scst_loss = -(advantages * seq_log_prob).mean()
        #     self.log("Train/Loss/SCST", scst_loss, sync_dist=True, on_step=True, prog_bar=True)

        #     loss += self.alpha_scst * scst_loss

        self.log("Train/Loss/MLE", ce_loss, sync_dist=True, on_step=True, prog_bar=True)
        self.log("Train/Loss/KL", kl_loss, sync_dist=True, on_step=True, prog_bar=True)
        self.log("Train/Loss/Align", align_loss, sync_dist=True, on_step=True, prog_bar=True)
        self.log("Train/Loss/Var", var_loss, sync_dist=True, on_step=True, prog_bar=True)
        self.log("Train/Acc/EM", em, sync_dist=True, on_step=True, prog_bar=True)
        self.log("Train/Acc/ES", es, sync_dist=True, on_step=True, prog_bar=True)
        self.log("Train/Acc/Precision", cum_prec, sync_dist=True, on_step=True, prog_bar=True)
        self.log("Train/Acc/WJI", wji, sync_dist=True, on_step=True, prog_bar=True)
        self.log("Train/Loss/SCST", scst_loss, sync_dist=True, on_step=True, prog_bar=True)
        self.log("Train/Loss/All", loss, sync_dist=True, on_step=True, prog_bar=True)

        return loss

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

        ### cumulative precision 1 / L \sum_{i=1}^L P@i
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

        import json
        record = {'pred': pred_text, 'gold': gold_text}
        with open('ast_cfc_java.jsonl', 'a') as f:
            f.write(json.dumps(record) + "\n")

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
            kl_loss = torch.tensor(0.0, device=input_ids.device)
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

                # proj_var = projections.var(dim=0).mean()
                # var_loss = F.relu(1e-4 - proj_var)

                # loss += self.alpha_align * align_loss + self.alpha_align / 10 * var_loss
                loss += self.alpha_align * align_loss

            if self.alpha_kl is not None and self.alpha_kl > 0.0:
                teacher_input_ids, teacher_labels, teacher_attention_mask = self.get_inputs_and_labels_fim(batch['teacher_input_ids'])
                teacher_logits = self(
                    input_ids=teacher_input_ids,
                    attention_mask=teacher_attention_mask).logits

                kl_loss = get_kl_loss(
                    teacher_logits=teacher_logits,
                    teacher_labels=teacher_labels,
                    student_logits=logits,
                    student_labels=labels,
                    temperature=self.kl_temperature,
                )

                loss += self.alpha_kl * kl_loss

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

            # if self.alpha_scst is not None and self.alpha_scst > .0:
            #     assert input_ids.shape[0] == 1, 'Change the logic below'
            #     prompt_len = input_ids[labels == -100].shape[0] + 1
            #     # ===== Baseline: greedy decode =====
            #     greedy_ids = self.generate(
            #         input_ids[:, :prompt_len],
            #         attention_mask=attention_mask[:, :prompt_len],
            #         structure_values=structure_ids,
            #         num_structure_tokens=num_structure_tokens,
            #         max_new_tokens=50, do_sample=False,
            #         pad_token_id=self.tokenizer.eos_token_id
            #     )
            #     baseline_rewards = self.cal_exact_match(greedy_ids[0, prompt_len:], labels[labels != -100])

            #     # ===== Sampled decode (exploration) =====
            #     sampled_ids = self.generate(
            #         input_ids[:, :prompt_len],
            #         attention_mask=attention_mask[:, :prompt_len],
            #         structure_values=structure_ids,
            #         num_structure_tokens=num_structure_tokens,
            #         max_new_tokens=50, do_sample=True, top_p=0.9, temperature=1.0,
            #         pad_token_id=self.tokenizer.eos_token_id
            #     )
            #     sampled_rewards = self.cal_exact_match(sampled_ids[0, prompt_len:], labels[labels != -100])

            #     # ---- Log probs of sampled tokens ----
            #     sampled_input_ids = sampled_ids[:, :-1]
            #     sampled_attention_mask = (sampled_input_ids != self.pad_token_id).long()
            #     sampled_logits = self(
            #         input_ids=sampled_input_ids,
            #         attention_mask=sampled_attention_mask,
            #         structure_values=structure_ids,
            #         num_structure_tokens=num_structure_tokens).logits
            #     log_probs = F.log_softmax(sampled_logits, dim=-1)

            #     gen_tokens = sampled_ids[:, prompt_len:]
            #     gen_logits = log_probs[:, prompt_len-1:, :]
            #     seq_log_probs = gen_logits.gather(2, gen_tokens.unsqueeze(-1)).squeeze(-1)
            #     seq_log_prob = seq_log_probs.sum(dim=1)

            #     # ---- SCST loss ----
            #     rewards = torch.tensor(sampled_rewards, device=seq_log_prob.device, dtype=torch.float)
            #     baselines = torch.tensor(baseline_rewards, device=seq_log_prob.device, dtype=torch.float)
            #     advantages = rewards - baselines
            #     scst_loss = -(advantages * seq_log_prob).mean()
            #     self.log("Val/Loss/SCST", scst_loss, sync_dist=True, on_epoch=True, prog_bar=True)

            #     loss += self.alpha_scst * scst_loss

        self.log("Val/Loss/MLE", ce_loss, sync_dist=True, on_epoch=True, prog_bar=True)
        self.log("Val/Loss/KL", kl_loss, sync_dist=True, on_epoch=True, prog_bar=True)
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
                "val_kl": kl_loss, 
                "val_em": em,
                "val_es": es, 
                "val_precision":cum_prec,
                "val_wji": wji,
                "val_scst": scst_loss, 
                "val_all": loss}

    def on_validation_epoch_end(self):

        # cleaning up memory
        torch.cuda.empty_cache()
        gc.collect()

    def on_after_backward(self):
        with torch.no_grad():
            # Compute gradient norm (L2 norm)
            total_norm = 0.0
            for p in self.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

            # Log gradient norm
            self.log('grad_norm', total_norm, on_step=True, on_epoch=False, prog_bar=True)

            total_norm = 0.0
            for p in self.parameters():
                if p.grad is not None:
                    total_norm += p.data.norm(2).item() ** 2
            self.log('weight_norm', total_norm ** 0.5, on_step=True)

    def configure_optimizers(self):
        decay_parameters = get_parameter_names(self.model, [torch.nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optim_groups = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if n in decay_parameters and p.requires_grad
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if n not in decay_parameters and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        # if self.model.injector is not None:
        #     optim_groups.append({
        #         "params": list(self.model.injector.parameters()),
        #         "weight_decay": 0.0})

        # optimizer = FusedAdam(optim_groups, lr=self.lr)
        optimizer = AdamW(optim_groups, lr=self.lr)

        if self.lr_scheduler_type == 'None':
            return optimizer
        if self.lr_scheduler_type == 'inv_sqrt':
            scheduler = get_inverse_sqrt_schedule(optimizer, num_warmup_steps=self.num_warmup_steps)
        elif self.lr_scheduler_type == 'linear':
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=self.num_warmup_steps,
                                                        num_training_steps=self.num_training_steps)
        elif self.lr_scheduler_type == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=self.num_warmup_steps,
                                                        num_training_steps=self.num_training_steps)
        else:
            raise ValueError('Unrecognized lr scheduler name: {}'.format(self.lr_scheduler_type))

        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]
