import argparse
import json
from transformers import (
    AutoTokenizer,
    RobertaTokenizer,
    RobertaConfig,
    AutoConfig,
)
from preprocess import AST
import torch
import torch.nn.functional as F
import math
from tqdm import tqdm

from modeling_llava_code import LlavaCodeConfig, LlavaCodeForConditionalGeneration
from eval_metric import compute_metric_stmt
from eval_metric_cceval import compute_metric_stmt_cceval

device = torch.device("cuda:0")


def tokenize_patches(tokens, patch_length, tokenizer):

    num_injection_tokens = math.ceil(len(tokens) / patch_length)

    tokens_ids = []
    for i in range(num_injection_tokens):
        patch = tokens[i * patch_length: (i + 1) * patch_length]
        patch_tokens = [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token] \
            + patch + [tokenizer.sep_token]
        patch_ids = tokenizer.convert_tokens_to_ids(patch_tokens)
        tokens_ids.extend(patch_ids)
    tokens_ids = torch.tensor(tokens_ids, dtype=torch.long)

    return tokens_ids, num_injection_tokens


def prepare_prompt(tokenizer,
                   structure_tokenizer,
                   left_cxt,
                   right_cxt=None,
                   crossfile_cxt=None):

    if args.data_prefix == 'code_cfc':

        structure_tokens = structure_tokenizer.tokenize(crossfile_cxt)
        patch_length = args.max_structure_length - 4  # 4 special tokens for unixcoder
        structure_ids, num_injection_tokens = tokenize_patches(structure_tokens, patch_length, structure_tokenizer)
        structure_ids = F.pad(structure_ids,
                              (0, args.max_structure_length * num_injection_tokens - len(structure_ids)),
                              value=structure_tokenizer.pad_token_id)

        left_cxt_truncated = tokenizer.decode(tokenizer.encode(left_cxt)[-(args.max_seq_length - args.gen_length - num_injection_tokens - args.right_context_length):])
        right_cxt_truncated = tokenizer.decode(tokenizer.encode(right_cxt)[:args.right_context_length])
        prompt = '<CODE_STRUCTURE>' * num_injection_tokens + f'<fim_prefix>{left_cxt_truncated}' + f'<fim_suffix>{right_cxt_truncated}<fim_middle>'

        return prompt, structure_ids, torch.tensor([num_injection_tokens])

    elif args.data_prefix == 'ast_cfc':

        # AST function ignores comments
        structure_tokens = AST(crossfile_cxt.replace('#', ''), 'python', structure_tokenizer)
        patch_length = args.max_structure_length - 4  # 4 special tokens for unixcoder
        structure_ids, num_injection_tokens = tokenize_patches(structure_tokens, patch_length, structure_tokenizer)
        structure_ids = F.pad(structure_ids,
                              (0, args.max_structure_length * num_injection_tokens - len(structure_ids)),
                              value=structure_tokenizer.pad_token_id)

        left_cxt_truncated = tokenizer.decode(tokenizer.encode(left_cxt)[-(args.max_seq_length - args.gen_length - num_injection_tokens - args.right_context_length):])
        right_cxt_truncated = tokenizer.decode(tokenizer.encode(right_cxt)[:args.right_context_length])
        prompt = '<CODE_STRUCTURE>' * num_injection_tokens + f'<fim_prefix>{left_cxt_truncated}' + f'<fim_suffix>{right_cxt_truncated}<fim_middle>'

        return prompt, structure_ids, torch.tensor([num_injection_tokens])

    elif args.data_prefix == 'default':

        left_cxt_truncated = tokenizer.decode(tokenizer.encode(left_cxt)[-(args.max_seq_length - args.gen_length - args.right_context_length):])
        right_cxt_truncated = tokenizer.decode(tokenizer.encode(right_cxt)[:args.right_context_length])
        prompt = f'<fim_prefix>{left_cxt_truncated}' + f'<fim_suffix>{right_cxt_truncated}<fim_middle>'

        return prompt, None, None

    elif args.data_prefix == 'default_cfc':

        assert crossfile_cxt is not None
        left_cxt_truncated = tokenizer.decode(tokenizer.encode(left_cxt)[-(args.max_seq_length - args.gen_length - args.right_context_length - args.cfc_seq_length):])
        right_cxt_truncated = tokenizer.decode(tokenizer.encode(right_cxt)[:args.right_context_length])
        crossfile_cxt_truncated = tokenizer.decode(tokenizer.encode('\n\n' + crossfile_cxt)[:args.cfc_seq_length])
        prompt = f'<fim_prefix>{left_cxt_truncated}<fim_suffix>{right_cxt_truncated}{crossfile_cxt_truncated}<fim_middle>'

    else:

        raise ValueError(f'Unrecognized data_prefix: {args.data_prefix}')


def build_dataset(args, code_tokenizer, ast_tokenizer):
    with open(args.prompt_file) as f:
        raw_data = [json.loads(line) for line in f.readlines()]

    data = []
    for entry in raw_data:

        left_cxt = entry["prompt"]
        right_cxt = entry["right_context"]
        crossfile_cxt = None
        if 'crossfile_context' in entry:
            crossfile_cxt = entry["crossfile_context"] if type(entry["crossfile_context"]) == str else entry["crossfile_context"]['text']

        entry['llm_prompt'], entry['structure_ids'], entry['num_structure_tokens'] = \
            prepare_prompt(code_tokenizer, ast_tokenizer,
                           left_cxt, right_cxt, crossfile_cxt)

        data.append(entry)

    return data


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--language", type=str, required=True, help="language name")
    parser.add_argument("--model_checkpoint", type=str)

    parser.add_argument("--prompt_file", type=str, default=None, help="file with a list of prompts")
    parser.add_argument("--gen_length", type=int, default=50, help="max length of generated token sequence")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="max length of prompt")
    parser.add_argument("--max_structure_length", type=int, default=512, help="max length of structure sequence")
    parser.add_argument(
        "--right_context_length",
        type=int,
        default=512,
        help="For model_type=codelm_leftright_context: Text sequence length corresponding to the right context"
    )
    parser.add_argument("--output_dir", type=str, default="output_dir", help="output directory to save predictions")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")

    # only compute metric
    parser.add_argument("--only_compute_metric", action="store_true", help="only compute metric")
    # for cceval metric
    parser.add_argument("--compute_cceval_metric", type=lambda x:bool(int(x)), help="use cceval metric")

    parser.add_argument("--data_prefix", type=str, help="Determines data preprocessing")

    parser.add_argument(
        "--task",
        choices=["line_completion", "api_completion", "function_completion"],
        default="line_completion",
        help="task name"
    )

    parser.add_argument('--config', type=str, help='path to args config')

    args = parser.parse_args()

    print('Input args:', args)

    code_tokenizer = AutoTokenizer.from_pretrained('bigcode/starcoderbase-1b', use_fast=False)
    code_tokenizer.add_tokens(['<CODE_STRUCTURE>'])
    if code_tokenizer.pad_token_id is None:
        code_tokenizer.pad_token_id = code_tokenizer.eos_token_id

    ast_tokenizer = RobertaTokenizer.from_pretrained("microsoft/unixcoder-base")
    ast_tokenizer.add_tokens(["<mask0>"], special_tokens=True)

    data = build_dataset(args, code_tokenizer, ast_tokenizer)

    structure_config = RobertaConfig.from_pretrained("microsoft/unixcoder-base")
    structure_config.model_id = "microsoft/unixcoder-base"
    text_config = AutoConfig.from_pretrained('bigcode/starcoderbase-1b')
    text_config.model_id = 'bigcode/starcoderbase-1b'
    text_config.vocab_size = text_config.vocab_size + 1  # for a new <CODE_STRUCTURE>
    configuration = LlavaCodeConfig(structure_config, text_config,
                                    pad_token_id=code_tokenizer.pad_token_id,
                                    structure_token_id=49152)

    if args.model_checkpoint:
        model = LlavaCodeForConditionalGeneration \
            .load_from_checkpoint(args.model_checkpoint, config=configuration).to(device)
    else:
        model = LlavaCodeForConditionalGeneration(configuration).to(device)

    all_preds = []
    for entry in tqdm(data):

        entropies = []
        with torch.no_grad():
            inputs = code_tokenizer(entry['llm_prompt'], return_tensors='pt').to(device)
            cut_at = inputs.input_ids.shape[1]
            if args.data_prefix not in ['default', 'default_cfc']:

                structure_ids = entry['structure_ids'].to(device)
                num_structure_tokens = entry['num_structure_tokens'].to(device)
                cur_pred = model.generate(**inputs,
                                          use_cache=True,
                                          do_sample=False,
                                          structure_values=structure_ids,
                                          num_structure_tokens=num_structure_tokens,
                                          max_new_tokens=args.gen_length,
                                          bad_words_ids=[[configuration.structure_token_id]])

            else:

                cur_pred = model.generate(**inputs,
                                          use_cache=True,
                                          do_sample=False,
                                          max_new_tokens=args.gen_length,
                                          bad_words_ids=[[configuration.structure_token_id]])

            if cur_pred[0, -1] == code_tokenizer.eos_token_id:
                prediction = code_tokenizer.decode(cur_pred[0][cut_at:-1])
            else:
                prediction = code_tokenizer.decode(cur_pred[0][cut_at:])

            all_preds.append({
                "task_id": entry["metadata"]["task_id"],
                "pred": prediction,
            })

    with open(f"{args.output_dir}/prediction.jsonl", "w", encoding="utf-8") as f_pred:
        for entry in all_preds:
            f_pred.write(json.dumps(entry) + "\n")

    if args.compute_cceval_metric:
        compute_metric_stmt_cceval(args)
    else:
        compute_metric_stmt(args)
