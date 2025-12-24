from transformers import (
    AutoConfig,
    PretrainedConfig,
    BitsAndBytesConfig,
    CONFIG_MAPPING
)


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
