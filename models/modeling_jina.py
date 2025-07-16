import torch.nn as nn


class JinaEncoder(nn.Module):
    """Wrapper Module for encoding inputs with Jina model
    """
    def __init__(self, model, config):
        super(JinaEncoder, self).__init__()
        self.model = model
        self.config = config

    def forward(self, input_ids):
        attn_mask = input_ids.ne(self.config.pad_token_id)
        out = self.model(input_ids=input_ids, attention_mask=attn_mask)

        return None, out.pooler_output
