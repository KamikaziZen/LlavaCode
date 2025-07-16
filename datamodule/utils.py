import torch


def pack_fim_inputs(fim_tokens_ids, left_context_ids, right_context_ids, target_ids, structure_token_id, num_structure_tokens):

    fim_prefix_id, fim_suffix_id, fim_middle_id = fim_tokens_ids.reshape(3, 1)  # else, fim tokens are 0-dim

    input_ids = torch.cat([
        fim_prefix_id,
        left_context_ids,
        fim_suffix_id,
        right_context_ids,
        torch.tensor([structure_token_id] * num_structure_tokens),
        fim_middle_id,
        target_ids]).to(torch.long)

    return input_ids
