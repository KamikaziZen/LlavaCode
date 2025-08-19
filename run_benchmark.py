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

from models import LlavaCodeConfig, LlavaCodeForConditionalGeneration
from eval_metric import compute_metric_stmt
from eval_metric_cceval import compute_metric_stmt_cceval
from datamodule import STRUCTURE_TOKEN, FIMMAP

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


def prepare_prompt(args,
                   tokenizer,
                   structure_tokenizer,
                   fim_tokens,
                   left_cxt,
                   right_cxt=None,
                   crossfile_cxt=None):
    """Dataset type: 10 chunks of cross-file context, 10 lines each, stored as an array
    """
    fim_prefix, fim_suffix, fim_middle = fim_tokens

    if args.data_prefix == 'code_cfc':
        # splitting context into chunks
        # one chunk for one file
        lines = crossfile_cxt.splitlines()[1:]  # removing the "Here are some examples..." line
        skip = False
        current_lines = []
        chunks = []
        for line in lines:
            if line.startswith('# the below code fragment can be found in:'):
                skip = True
                if current_lines:
                    chunks.append('\n'.join(current_lines))
                current_lines = []
            elif skip:  # skipping the file path
                skip = False
            elif line:
                current_lines.append(line.strip())
        if current_lines:
            chunks.append('\n'.join(current_lines))

        # restrict number of injection tokens (RAG files)
        num_injection_tokens = min(args.num_structure_tokens, len(chunks))
        chunks = chunks[:num_injection_tokens]

        structure_ids = []
        for cfc in chunks:
            cfc_tokens = structure_tokenizer.tokenize(cfc)
            cfc_tokens = cfc_tokens[:args.max_structure_length - 4]
            # cfc_tokens = cfc_tokens[:args.max_structure_length - 4]  # 4 special tokens for unixcoder
            # cfc_tokens = [structure_tokenizer.cls_token, "<encoder-only>", structure_tokenizer.sep_token] \
            #     + cfc_tokens + [structure_tokenizer.sep_token]
            chunk_ids = structure_tokenizer.convert_tokens_to_ids(cfc_tokens)
            structure_ids.extend(F.pad(torch.tensor(chunk_ids), (0, args.max_structure_length-len(chunk_ids)), value=structure_tokenizer.pad_token_id))
        structure_ids = torch.tensor(structure_ids, dtype=torch.long)

        left_cxt_truncated = tokenizer.decode(tokenizer.encode(left_cxt)[-(args.max_seq_length - args.gen_length - num_injection_tokens - args.right_context_length):])
        right_cxt_truncated = tokenizer.decode(tokenizer.encode(right_cxt)[:args.right_context_length])
        prompt = f"{fim_prefix}{left_cxt_truncated}{fim_suffix}{right_cxt_truncated}# Here are some relevant code fragments from other files of the repo:{'<CODE_STRUCTURE>' * num_injection_tokens}{fim_middle}"
        # prompt = f"# Here are some relevant code fragments from other files of the repo:{'<CODE_STRUCTURE>' * num_injection_tokens}{fim_prefix}{left_cxt_truncated}{fim_suffix}{right_cxt_truncated}{fim_middle}"

        return prompt, structure_ids, torch.tensor([num_injection_tokens])

    elif args.data_prefix == 'ast_cfc':
        # splitting context into chunks
        # one chunk for one file
        lines = crossfile_cxt.splitlines()[1:]  # removing the "Here are some examples..." line
        skip = False
        current_lines = []
        chunks = []
        for line in lines:
            if line.startswith('# the below code fragment can be found in:'):
                skip = True
                if current_lines:
                    chunks.append('\n'.join(current_lines))
                current_lines = []
            elif skip:  # skipping the file path
                skip = False
            elif line:
                current_lines.append(line.strip())
        if current_lines:
            chunks.append('\n'.join(current_lines))

        # restrict number of injection tokens (RAG files)
        num_injection_tokens = min(args.num_structure_tokens, len(chunks))
        chunks = chunks[:num_injection_tokens]

        structure_ids = []
        for cfc in chunks:
            ast_tokens = AST(cfc.replace('#', ''), 'python', structure_tokenizer)  # decommenting
            ast_tokens = ast_tokens[:args.max_structure_length - 4]  # 4 special tokens for unixcoder
            chunk_tokens = [structure_tokenizer.cls_token, "<encoder-only>", structure_tokenizer.sep_token] \
                + ast_tokens + [structure_tokenizer.sep_token]
            chunk_ids = structure_tokenizer.convert_tokens_to_ids(chunk_tokens)
            structure_ids.extend(F.pad(torch.tensor(chunk_ids), (0, args.max_structure_length-len(chunk_ids)), value=structure_tokenizer.pad_token_id))
        structure_ids = torch.tensor(structure_ids, dtype=torch.long)

        left_cxt_truncated = tokenizer.decode(tokenizer.encode(left_cxt)[-(args.max_seq_length - args.gen_length - num_injection_tokens - args.right_context_length):])
        right_cxt_truncated = tokenizer.decode(tokenizer.encode(right_cxt)[:args.right_context_length])
        prompt = f"{fim_prefix}{left_cxt_truncated}{fim_suffix}{right_cxt_truncated}{'<CODE_STRUCTURE>' * num_injection_tokens}{fim_middle}"

        return prompt, structure_ids, torch.tensor([num_injection_tokens])

    # elif args.data_prefix == 'codeast_cfc':

    #     structure_tokens = structure_tokenizer.tokenize(crossfile_cxt)
    #     # AST function ignores comments
    #     structure_tokens += AST(crossfile_cxt.replace('#', ''), 'python', structure_tokenizer)
    #     patch_length = args.max_structure_length - 4  # 4 special tokens for unixcoder
    #     structure_ids, num_injection_tokens = tokenize_patches(structure_tokens, patch_length, structure_tokenizer)
    #     structure_ids = F.pad(structure_ids,
    #                           (0, args.max_structure_length * num_injection_tokens - len(structure_ids)),
    #                           value=structure_tokenizer.pad_token_id)

    #     left_cxt_truncated = tokenizer.decode(tokenizer.encode(left_cxt)[-(args.max_seq_length - args.gen_length - num_injection_tokens - args.right_context_length):])
    #     right_cxt_truncated = tokenizer.decode(tokenizer.encode(right_cxt)[:args.right_context_length])
    #     prompt = f'<fim_prefix>{left_cxt_truncated}' + f'<fim_suffix>{right_cxt_truncated}' + '<CODE_STRUCTURE>' * num_injection_tokens + '<fim_middle>'

    #     return prompt, structure_ids, torch.tensor([num_injection_tokens])

    elif args.data_prefix == 'default':

        left_cxt_truncated = tokenizer.decode(tokenizer.encode(left_cxt)[-(args.max_seq_length - args.gen_length - args.right_context_length):])
        right_cxt_truncated = tokenizer.decode(tokenizer.encode(right_cxt)[:args.right_context_length])
        prompt = f"{fim_prefix}{left_cxt_truncated}{fim_suffix}{right_cxt_truncated}{fim_middle}"

        return prompt, None, None

    elif args.data_prefix == 'default_cfc':

        assert crossfile_cxt is not None

        lines = crossfile_cxt.splitlines()
        skip = False
        current_lines = []
        chunks = []
        for line in lines[1:]:
            if line.startswith('# the below code fragment can be found in:'):
                skip = True
                if current_lines:
                    chunks.append('\n'.join(current_lines))
                current_lines = [line.strip()]
            elif skip:  # skipping the file path
                skip = False
                current_lines.append(line.strip())
            elif line:
                current_lines.append(line.strip())
        if current_lines:
            chunks.append('\n'.join(current_lines))
        # restrict number of injection tokens (RAG files)
        num_injection_tokens = min(args.num_structure_tokens, len(chunks))
        chunks = chunks[:num_injection_tokens]
        crossfile_cxt = '\n\n'.join(lines[:1] + chunks)

        left_cxt_truncated = tokenizer.decode(tokenizer.encode(left_cxt)[-(args.max_seq_length - args.gen_length - args.right_context_length - args.cfc_seq_length):])
        right_cxt_truncated = tokenizer.decode(tokenizer.encode(right_cxt)[:args.right_context_length])
        crossfile_cxt_truncated = tokenizer.decode(tokenizer.encode('\n\n' + crossfile_cxt)[:args.cfc_seq_length])
        if 'starcoder' in args.text_model_id.lower():
            prompt = f'{fim_prefix}{left_cxt_truncated}{fim_suffix}{right_cxt_truncated}{crossfile_cxt_truncated}{fim_middle}'
        elif 'qwen' in args.text_model_id.lower():
            prompt = f'{crossfile_cxt_truncated}{fim_prefix}{left_cxt_truncated}{fim_suffix}{right_cxt_truncated}{fim_middle}'
            # prompt = f'{fim_prefix}{left_cxt_truncated}{fim_suffix}{right_cxt_truncated}{crossfile_cxt_truncated}{fim_middle}'

        return prompt, None, None

    else:

        raise ValueError(f'Unrecognized data_prefix: {args.data_prefix}')


def build_dataset(args, code_tokenizer, ast_tokenizer, fim_tokens):
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
            prepare_prompt(
                args, code_tokenizer, ast_tokenizer, fim_tokens, left_cxt, right_cxt, crossfile_cxt)

        data.append(entry)

    return data


def remove_tokens(s, tokens=["<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|fim_pad|>", "<|repo_name|>", "<|file_sep|>", "<|im_start|>", "<|im_end|>"]):
    indexes = [s.find(token) for token in tokens]
    valid_indexes = [idx for idx in indexes if idx != -1]
    if valid_indexes:
        return s[:min(valid_indexes)]
    else:
        return s


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--text_model_id", type=str, required=True)
    parser.add_argument("--structure_model_id", type=str, required=True)
    parser.add_argument("--language", type=str, required=True, help="language name")
    parser.add_argument("--model_checkpoint", type=str)
    parser.add_argument("--projector_checkpoint", type=str)
    parser.add_argument("--task", type=str, choices=["line_completion", "api_completion", "function_completion"])
    parser.add_argument("--prompt_file", type=str, default=None, help="file with a list of prompts")
    parser.add_argument("--gen_length", type=int, default=50, help="max length of generated token sequence")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="max length of prompt")
    parser.add_argument("--max_structure_length", type=int, default=512, help="max length of structure sequence")
    parser.add_argument("--right_context_length",
                        type=int,
                        default=512,
                        help="For model_type=codelm_leftright_context: Text sequence length corresponding to the right context")
    parser.add_argument("--cfc_seq_length",
                        type=int,
                        default=512,
                        help="For model_type=codelm_cfc: Text sequence length corresponding to the retrieved nodes")
    parser.add_argument("--num_structure_tokens", type=int, required=True, help='number of embeddings reserved for RAG injection')
    parser.add_argument("--output_dir", type=str, default="output_dir", help="output directory to save predictions")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")
    parser.add_argument("--only_compute_metric", action="store_true", help="only compute metric")
    parser.add_argument("--compute_cceval_metric", type=lambda x: bool(int(x)), help="use cceval metric")
    parser.add_argument("--data_prefix", type=str, help="Determines data preprocessing")
    parser.add_argument('--config', type=str, help='path to args config')

    args = parser.parse_args()
    print('Input args:', args)

    code_tokenizer = AutoTokenizer.from_pretrained(args.text_model_id, use_fast=False)
    code_tokenizer.add_tokens(['<CODE_STRUCTURE>'])
    if code_tokenizer.pad_token_id is None:  # case with starcoder
        code_tokenizer.pad_token_id = code_tokenizer.eos_token_id
    structure_token_id = code_tokenizer.convert_tokens_to_ids(STRUCTURE_TOKEN)

    if 'unixcoder' in args.structure_model_id.lower():
        structure_config = RobertaConfig.from_pretrained(args.structure_model_id)
        structure_tokenizer = RobertaTokenizer.from_pretrained(args.structure_model_id)
        structure_tokenizer.add_tokens(["<mask0>"], special_tokens=True)
    elif 'graphcodebert' in args.structure_model_id.lower():
        structure_config = RobertaConfig.from_pretrained(args.structure_model_id)
        structure_tokenizer = RobertaTokenizer.from_pretrained(args.structure_model_id)
    elif 'jina' in args.structure_model_id.lower():
        structure_config = AutoConfig.from_pretrained(args.structure_model_id)
        structure_tokenizer = AutoTokenizer.from_pretrained(args.structure_model_id)
    else:
        raise NotImplementedError(args.structure_model_id)
    structure_config.model_id = args.structure_model_id
    text_config = AutoConfig.from_pretrained(args.text_model_id)
    text_config.model_id = args.text_model_id

    # TODO: is it possible to include <CODE_STRUCTURE> -> vector mapping without resizing embeddings?
    # possible implementation: qwen tokens <|repo_name|> and <|file_sep|> tokens
    # this is necessary for resize_token_embeddings() call
    text_config.vocab_size = text_config.vocab_size + 1  # for a new <CODE_STRUCTURE>
    configuration = LlavaCodeConfig(structure_config, text_config,
                                    pad_token_id=code_tokenizer.pad_token_id,
                                    structure_token_id=structure_token_id)
    print('tokenizer shapes:', code_tokenizer.vocab_size, len(code_tokenizer))  # delete later

    if args.model_checkpoint:
        print(f'Loading model from checkpoint: {args.model_checkpoint}')
        model = LlavaCodeForConditionalGeneration \
            .load_from_checkpoint(args.model_checkpoint, config=configuration).to(device)
        print(f'after checkpoint: {model.model.multi_modal_projector.linear_1.weight.data.norm(2)}')
    else:
        model = LlavaCodeForConditionalGeneration(configuration).to(device)
    if args.projector_checkpoint:
        print(f'Loading projection weighs from {args.projector_checkpoint}')
        model.multi_modal_projector.load_state_dict(torch.load(args.projector_checkpoint))
    model.eval()

    if 'qwen' in args.text_model_id.lower():
        fim_tokens = FIMMAP['qwen2.5']
    elif 'starcoder' in args.text_model_id.lower():
        fim_tokens = FIMMAP['starcoder']
    else:
        raise NotImplementedError('No such model in FIM mapping')
    print('fim tokens:', fim_tokens)
    data = build_dataset(args, code_tokenizer, structure_tokenizer, fim_tokens)

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

            prediction = code_tokenizer.decode(cur_pred[0][cut_at:], skip_special_tokens=True)

            # <|fim_pad|>, <|file_sep|>, <|fim_prefix|> are not removed by skip_special_tokens=True, manual removal
            if 'qwen' in args.text_model_id.lower():
                prediction = remove_tokens(prediction)

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
