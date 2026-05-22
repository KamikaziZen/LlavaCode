import torch
from .const import FIMMAP


def get_fim_tokens(model_id):
    if 'qwen2.5' in model_id.lower():
        fim_tokens = FIMMAP['qwen2.5']
    elif 'starcoder' in model_id.lower():
        fim_tokens = FIMMAP['starcoder']
    else:
        raise NotImplementedError(f'No such model in FIM mapping: {model_id}')

    return fim_tokens


def truncate_trash(s, markers=(
    # End-of-completion / document-boundary markers the frozen LM emits to signal
    # "completion done". They are only visible if the generation was decoded with
    # skip_special_tokens=False; everything from the first marker onward is
    # over-generation (a new FIM block / next file) and must be dropped, otherwise
    # skip_special_tokens silently glues it onto the answer
    # (e.g. 'return image' -> 'return imagee_size,').
    # StarCoder2 family
    "<file_sep>", "<fim_prefix>", "<fim_suffix>", "<fim_middle>", "<fim_pad>",
    "<repo_name>", "<empty_output>",
    # Qwen2.5-Coder family
    "<|file_sep|>", "<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>", "<|fim_pad|>",
    "<|repo_name|>", "<|im_end|>",
    # shared end-of-text
    "<|endoftext|>",
)):
    indexes = [s.find(marker) for marker in markers]
    valid_indexes = [idx for idx in indexes if idx != -1]
    if valid_indexes:
        return s[:min(valid_indexes)]
    else:
        return s