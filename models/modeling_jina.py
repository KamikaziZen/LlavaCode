import torch
import torch.nn as nn
import torch.nn.functional as F


class JinaEncoder(nn.Module):
    """Wrapper Module for encoding inputs with Jina model
    """
    def __init__(self, model, config):
        super(JinaEncoder, self).__init__()
        self.model = model
        self.config = config

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # last hidden state
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * mask_expanded, dim=1) / torch.clamp(mask_expanded.sum(1), min=1e-9)

    def forward(self, input_ids):
        attn_mask = input_ids.ne(self.config.pad_token_id)
        out = self.model(input_ids=input_ids, attention_mask=attn_mask)
        embeddings = self.mean_pooling(out, attn_mask)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return None, embeddings
