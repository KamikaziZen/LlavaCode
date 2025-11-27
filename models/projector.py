import torch.nn as nn
from .config import LlavaCodeConfig
from transformers.activations import ACT2FN


class LlavaCodeMultiModalProjector4L(nn.Module):
    def __init__(self, config: LlavaCodeConfig):
        super().__init__()
        self.linear_1 = nn.Linear(
            config.structure_config.hidden_size,
            config.text_config.hidden_size * 3,
            bias=config.multimodal_projector_bias,
        )
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(
            config.text_config.hidden_size * 3, config.text_config.hidden_size * 4, bias=config.multimodal_projector_bias
        )
        self.linear_3 = nn.Linear(
            config.text_config.hidden_size * 4, config.text_config.hidden_size * 2, bias=config.multimodal_projector_bias
        )
        self.linear_4 = nn.Linear(
            config.text_config.hidden_size * 2, config.text_config.hidden_size, bias=config.multimodal_projector_bias
        )
        self.ln_1 = nn.LayerNorm(config.text_config.hidden_size * 3)
        self.ln_2 = nn.LayerNorm(config.text_config.hidden_size * 4)
        self.ln_3 = nn.LayerNorm(config.text_config.hidden_size * 2)

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
        hidden_states = self.act(hidden_states)
        hidden_states = self.ln_3(hidden_states)
        hidden_states = self.linear_4(hidden_states)
        return hidden_states


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


class LlavaCodeMultiModalProjector2L(nn.Module):
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
        self.ln_1 = nn.LayerNorm(config.text_config.hidden_size)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def forward(self, structure_features):
        hidden_states = self.linear_1(structure_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.ln_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states