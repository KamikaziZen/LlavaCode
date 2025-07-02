from transformers import (
    PreTrainedModel,
    LlamaConfig,
    CLIPVisionConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    PretrainedConfig,
    GenerationMixin,
    CONFIG_MAPPING
)

from transformers.utils import can_return_tuple, LossKwargs
from transformers.processing_utils import Unpack
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers.activations import ACT2FN
from transformers.trainer_pt_utils import get_parameter_names
from transformers.optimization import (
    get_linear_schedule_with_warmup, 
    get_inverse_sqrt_schedule,
    get_cosine_schedule_with_warmup
)

from pytorch_lightning import LightningModule

import torch
import torch.nn as nn
from torch.optim import AdamW
import math

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from unixcoder import UniXcoder


class LlavaCodeConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LlavaNextForConditionalGeneration`]. It is used to instantiate an
    Llava-NeXT model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the [llava-hf/llava-v1.6-mistral-7b-hf](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf)
    model.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        structure_config (`Union[AutoConfig, dict]`,  *optional*, defaults to `CLIPVisionConfig`):
            The config object or dictionary of the structure backbone.
        text_config (`Union[AutoConfig, dict]`, *optional*, defaults to `LlamaConfig`):
            The config object or dictionary of the text backbone.
        structure_token_index (`int`, *optional*, defaults to 25782):
            The structure token index to encode the structure prompt.
        projector_hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The activation function used by the multimodal projector.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        multimodal_projector_bias (`bool`, *optional*, defaults to `True`):
            Whether to use bias in the multimodal projector.

    Example:

    ```python
    >>> from transformers import LlavaNextForConditionalGeneration, LlavaNextConfig, CLIPVisionConfig, LlamaConfig

    >>> # Initializing a Llama config
    >>> text_config = LlamaConfig()

    >>> # Initializing a Llava-Next llava-hf/llava-v1.6-mistral-7b-hf style configuration
    >>> configuration = LlavaNextConfig(vision_config, text_config)

    >>> # Initializing a model from the llava-hf/llava-v1.6-mistral-7b-hf style configuration
    >>> model = LlavaNextForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

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
        **kwargs,
    ):
        self.projector_hidden_act = projector_hidden_act
        self.multimodal_projector_bias = multimodal_projector_bias

        self.structure_config = structure_config

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "llama"
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["llama"]()

        self.text_config = text_config

        self.structure_token_id = structure_token_id
        self.pad_token_id = pad_token_id

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class LlavaCodeMultiModalProjector(nn.Module):
    def __init__(self, config: LlavaCodeConfig):
        super().__init__()
        self.linear_1 = nn.Linear(
            config.structure_config.hidden_size,
            config.text_config.hidden_size,
            bias=config.multimodal_projector_bias,
        )
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(
            config.text_config.hidden_size, config.text_config.hidden_size, bias=config.multimodal_projector_bias
        )

    def forward(self, image_features):
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


@dataclass
class LlavaCodeModelOutputWithPast(BaseModelOutputWithPast):
    """
    Base class for Llava outputs, with hidden states and attentions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
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
            A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
            structure_hidden_states of the model produced by the structure encoder and after projecting the last hidden state.
    """

    structure_hidden_states: Optional[torch.FloatTensor] = None

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

class LlavaCodeModel(LlavaCodePreTrainedModel):
    _checkpoint_conversion_mapping = {"language_model.model": "language_model"}

    def __init__(self, config: LlavaCodeConfig):
        super().__init__(config)
        self.structure_model = UniXcoder(self.config.structure_config.model_id)

        self.multi_modal_projector = LlavaCodeMultiModalProjector(config)
        embed_std = 1 / math.sqrt(config.text_config.hidden_size)

        self.vocab_size = config.text_config.vocab_size
        self.language_model = AutoModel.from_pretrained(self.config.text_config.model_id)
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_structure_features(self, structure_values, nums_structure_tokens):
        # structure values: ast tree sequence ids
        # nums_structure_tokens: number of structure tokens for each sample in a batch
        max_num = nums_structure_tokens.max()
        row_ids = torch.arange(max_num).expand(len(nums_structure_tokens), max_num).to(nums_structure_tokens.device)
        mask = row_ids < nums_structure_tokens.unsqueeze(1)
        flat_mask = mask.flatten()
        _, ast_embedding = self.structure_model(structure_values.reshape(-1, 512))
        # taking only those features that correspond to code_structure tokens
        # others fully consist of padding
        ast_embedding = ast_embedding[flat_mask]

        structure_features = self.multi_modal_projector(ast_embedding)
        return structure_features

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        structure_values: torch.LongTensor = None,
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
    ) -> Union[Tuple, LlavaCodeModelOutputWithPast]:
        r"""
        
        """
        # checking if cache is already in use (use_cache=True and iter > 1)
        using_cache = isinstance(past_key_values, list)
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids) # from language model only

        structure_features = None
        if structure_values is not None and not using_cache:
            structure_features = self.get_structure_features(structure_values, num_structure_tokens)
            special_structure_mask = (input_ids == self.config.structure_token_id).unsqueeze(-1)
            special_structure_mask = special_structure_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
            assert inputs_embeds[special_structure_mask].numel() == structure_features.numel(), \
                f'Mask does not correspond to the number of structure features: {inputs_embeds[special_structure_mask].numel()} != {structure_features.numel()}'
            structure_features = structure_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_structure_mask, structure_features)

        outputs = self.language_model(
            # input_ids: Optional[torch.Tensor] = None,
            attention_mask=attention_mask,
            # token_type_ids: Optional[torch.Tensor] = None,
            position_ids=position_ids,
            past_key_values=past_key_values if using_cache else None,  # gpt-related kostyl
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

        return LlavaCodeModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            structure_hidden_states=structure_features,
        )


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
    structure_hidden_states: Optional[torch.FloatTensor] = None


class LlavaCodeForConditionalGeneration(LlavaCodePreTrainedModel, GenerationMixin, LightningModule):
    _checkpoint_conversion_mapping = {
        "^language_model.model": "model.language_model",
        "^structure_model": "model.structure_model",
        "^multi_modal_projector": "model.multi_modal_projector",
        "^language_model.lm_head": "lm_head",
    }
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: LlavaCodeConfig):
        super().__init__(config)
        self.model = LlavaCodeModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        self.mle_loss = torch.nn.CrossEntropyLoss()
        self.vocab_size = self.config.text_config.vocab_size
        self.language_model.resize_token_embeddings(self.vocab_size)

        self.post_init()

    def set_trainer_args(self, trainer_args):
        self.trainer_args = trainer_args

    def setup(self, stage):
        if stage == 'fit':
            # Hyperparamters and Configuration
            self.num_nodes = self.trainer_args.num_nodes
            self.dropout_p = self.trainer_args.dropout_p
            self.functional_dropout = self.trainer_args.functional_dropout
            self.pad_token_id = self.trainer_args.pad_token_id

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
            # Loss Configuration
            self.loss = self.trainer_args.loss
            assert self.loss in ["MLE_Only", "ContraCLM", "ContraCLMTok", "ContraCLMSeq", "Repoformer"], \
                f"Loss: `{self.loss}` is not supported!"

            self.validation_step_outputs = []

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

    # Make modules available throught conditional class for BC
    @property
    def language_model(self):
        return self.model.language_model

    @property
    def vision_tower(self):
        return self.model.vision_tower

    @property
    def multi_modal_projector(self):
        return self.model.multi_modal_projector

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        structure_values: torch.FloatTensor = None,
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

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.vocab_size, **kwargs
            )

        return LlavaCodeCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            structure_hidden_states=outputs.structure_hidden_states,
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

    def get_inputs_and_labels(self, token_ids):
        inp_tensor = token_ids[:, :-1].clone()

        lbl_tensor = token_ids[:, 1:].clone()
        lbl_tensor[lbl_tensor[:, :] == self.pad_token_id] = -100

        attention_mask = torch.ones_like(inp_tensor)
        attention_mask = attention_mask.masked_fill(inp_tensor.eq(self.pad_token_id), 0.0).type(torch.bool)

        return inp_tensor, lbl_tensor, attention_mask

    def training_step(self, batch, batch_idx):
        token_ids, ast_ids, num_structure_tokens = batch['input_ids'], batch['ast_ids'], batch['num_structure_tokens']
        input_ids, labels, attention_mask = self.get_inputs_and_labels(token_ids)
        # first forward pass
        logits = self(input_ids=input_ids,
                      attention_mask=attention_mask,
                      structure_values=ast_ids,
                      num_structure_tokens=num_structure_tokens).logits

        loss = self.mle_loss(logits.view(-1, self.vocab_size), labels.view(-1))
        self.log("Train/Loss/MLE", loss, sync_dist=True, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        eval_fct = torch.nn.CrossEntropyLoss()
        token_ids, ast_ids, num_structure_tokens = batch['input_ids'], batch['ast_ids'], batch['num_structure_tokens']
        input_ids, labels, attention_mask = self.get_inputs_and_labels(token_ids)
        logits = self(input_ids=input_ids,
                      attention_mask=attention_mask,
                      structure_values=ast_ids,
                      num_structure_tokens=num_structure_tokens).logits
        loss = eval_fct(logits.view(-1, self.vocab_size), labels.view(-1))
        self.validation_step_outputs.append(loss)
        return loss

    def on_validation_epoch_end(self):
        val_loss = torch.stack(self.validation_step_outputs).mean()
        perplexity = torch.exp(val_loss)
        self.log("Valid/Loss/MLE", val_loss, sync_dist=True, on_epoch=True, prog_bar=True)
        self.log("Valid/Loss/Perplexity", perplexity, sync_dist=True, on_epoch=True, prog_bar=True)
        self.validation_step_outputs.clear()  # free memory

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
