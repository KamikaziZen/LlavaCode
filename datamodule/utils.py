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


def truncate_trash(s, markers=["<|file_sep|><|fim_prefix|>", "<|fim_pad|>"]):
    indexes = [s.find(marker) for marker in markers]
    valid_indexes = [idx for idx in indexes if idx != -1]
    if valid_indexes:
        return s[:min(valid_indexes)]
    else:
        return s