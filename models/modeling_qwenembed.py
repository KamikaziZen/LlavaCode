import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class QwenEmbedEncoder(nn.Module):
    """Wrapper Module for encoding inputs with Qwen3 Embedding model
    """
    def __init__(self, model, config):
        super(QwenEmbedEncoder, self).__init__()
        self.model = model
        self.config = config

    @property
    def device(self):
        return self.model.device

    def last_token_pool(self, last_hidden_states: Tensor, attention_mask: Tensor, left_padding=True) -> Tensor:
        # left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def forward(self, input_ids):
        attn_mask = input_ids.ne(self.config.pad_token_id)
        outputs = self.model(input_ids=input_ids, attention_mask=attn_mask)
        embeddings = self.last_token_pool(outputs.last_hidden_state, attn_mask, left_padding=False)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return None, embeddings
