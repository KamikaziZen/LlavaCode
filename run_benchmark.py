import argparse
import json
import os 
from transformers import (
    AutoTokenizer,
    RobertaTokenizer,
    RobertaConfig,
    AutoConfig,
)
from preprocess import AST
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import math
from tqdm import tqdm
import re

from models import LlavaCodeConfig, LlavaCodeForConditionalGeneration
from eval_metric import compute_metric_stmt
from eval_metric_cceval import compute_metric_stmt_cceval
from datamodule.const import STRUCTURE_TOKEN
from datamodule.utils import truncate_trash, get_fim_tokens


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

    if args.data_prefix == 'code_cfc_uxc':
        assert False
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
        num_structure_tokens = min(args.num_structure_tokens, len(chunks))
        chunks = chunks[:num_structure_tokens]

        structure_ids = []
        for cfc in chunks:
            cfc_tokens = structure_tokenizer.tokenize(cfc)
            cfc_tokens = cfc_tokens[:args.max_structure_length - 4]  # 4 special tokens for unixcoder
            cfc_tokens = [structure_tokenizer.cls_token, "<encoder-only>", structure_tokenizer.sep_token] \
                + cfc_tokens + [structure_tokenizer.sep_token]
            chunk_ids = structure_tokenizer.convert_tokens_to_ids(cfc_tokens)
            structure_ids.extend(F.pad(torch.tensor(chunk_ids), (0, args.max_structure_length-len(chunk_ids)), value=structure_tokenizer.pad_token_id))
        structure_ids = torch.tensor(structure_ids, dtype=torch.long)

        left_cxt_truncated = tokenizer.decode(tokenizer.encode(left_cxt)[-(args.max_seq_length - args.gen_length - num_structure_tokens - args.right_context_length):])
        right_cxt_truncated = tokenizer.decode(tokenizer.encode(right_cxt)[:args.right_context_length])
        prompt = f"{fim_prefix}{left_cxt_truncated}{fim_suffix}{right_cxt_truncated}{STRUCTURE_TOKEN * num_structure_tokens}{fim_middle}"
        # prompt = f"# Here are some relevant code fragments from other files of the repo:{'<CODE_STRUCTURE>' * num_structure_tokens}{fim_prefix}{left_cxt_truncated}{fim_suffix}{right_cxt_truncated}{fim_middle}"

        return prompt, structure_ids, torch.tensor([num_structure_tokens])

    elif args.data_prefix in ['code_cfc_jina', 'code_cfc_qwen']:
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
                current_lines.append(line)
        if current_lines:
            chunks.append('\n'.join(current_lines))

        # restrict number of injection tokens (RAG files)
        num_structure_tokens = min(args.num_structure_tokens, len(chunks))
        chunks = chunks[:num_structure_tokens]

        structure_ids = torch.empty(0, dtype=torch.long)
        for cfc in chunks:
            cfc_ids = structure_tokenizer(cfc, return_tensors='pt', truncation=True, max_length=args.max_structure_length).input_ids[0]
            structure_ids = torch.hstack([structure_ids, F.pad(cfc_ids, (0, args.max_structure_length-len(cfc_ids)), value=structure_tokenizer.pad_token_id)])

        lr_budget = args.max_seq_length - args.gen_length - 3  # 3 tokens for FIM
        rc_budget = int(lr_budget / (args.lc_rc_ratio + 1))
        lc_budget = int(rc_budget * args.lc_rc_ratio)

        left_cxt_truncated = tokenizer.decode(tokenizer.encode(left_cxt)[-lc_budget:])
        right_cxt_truncated = tokenizer.decode(tokenizer.encode(right_cxt)[:rc_budget])

        prompt = f"{fim_prefix}{left_cxt_truncated}{fim_suffix}{right_cxt_truncated}{STRUCTURE_TOKEN * num_structure_tokens}{fim_middle}"

        return prompt, structure_ids, torch.tensor([num_structure_tokens])

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
        num_structure_tokens = min(args.num_structure_tokens, len(chunks))
        chunks = chunks[:num_structure_tokens]

        structure_ids = torch.empty(0, dtype=torch.long)
        for cfc in chunks:
            ast_tokens = AST(cfc.replace('#', ''), 'python', structure_tokenizer)  # decommenting
            ast_tokens = ast_tokens[:args.max_structure_length - 4]  # 4 special tokens for unixcoder
            chunk_tokens = [structure_tokenizer.cls_token, "<encoder-only>", structure_tokenizer.sep_token] \
                + ast_tokens + [structure_tokenizer.sep_token]
            chunk_ids = structure_tokenizer.convert_tokens_to_ids(chunk_tokens)
            structure_ids = torch.hstack([structure_ids, F.pad(torch.tensor(chunk_ids), (0, args.max_structure_length-len(chunk_ids)), value=structure_tokenizer.pad_token_id)])

        left_cxt_truncated = tokenizer.decode(tokenizer.encode(left_cxt)[-(args.max_seq_length - args.gen_length - num_structure_tokens - args.right_context_length):])
        right_cxt_truncated = tokenizer.decode(tokenizer.encode(right_cxt)[:args.right_context_length])
        prompt = f"{fim_prefix}{left_cxt_truncated}{fim_suffix}{right_cxt_truncated}{STRUCTURE_TOKEN * num_structure_tokens}{fim_middle}"

        return prompt, structure_ids, torch.tensor([num_structure_tokens])

    elif args.data_prefix == 'default':

        lr_budget = args.max_seq_length - args.gen_length - 3 # 3 tokens for FIM
        rc_budget = int(lr_budget / (args.lc_rc_ratio + 1))
        lc_budget = int(rc_budget * args.lc_rc_ratio)

        left_cxt_truncated = tokenizer.decode(tokenizer.encode(left_cxt)[-lc_budget:])
        right_cxt_truncated = tokenizer.decode(tokenizer.encode(right_cxt)[:rc_budget])

        prompt = f"{fim_prefix}{left_cxt_truncated}{fim_suffix}{right_cxt_truncated}{fim_middle}"

        return prompt, None, None

    elif args.data_prefix == 'default_cfc':

        assert crossfile_cxt is not None

        # making the same lr_budget as for other experiments, not considering cfc length
        lr_budget = args.max_seq_length - args.gen_length - 3 # 3 tokens for FIM
        rc_budget = int(lr_budget / (args.lc_rc_ratio + 1))
        lc_budget = int(rc_budget * args.lc_rc_ratio)

        left_cxt_truncated = tokenizer.decode(tokenizer.encode(left_cxt)[-lc_budget:])
        right_cxt_truncated = tokenizer.decode(tokenizer.encode(right_cxt)[:rc_budget])
        crossfile_cxt_truncated = tokenizer.decode(tokenizer.encode(crossfile_cxt)[:args.cfc_seq_length])

        prompt = f'{fim_prefix}{left_cxt_truncated}{fim_suffix}{right_cxt_truncated}{crossfile_cxt_truncated}{fim_middle}'

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


def parse_args():
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
    parser.add_argument("--do_sample", action='store_true')
    parser.add_argument("--lc_rc_ratio", default=2.0)

    args = parser.parse_args()
    return args


def main_worker(rank, world_size, model, tokenizer, data, args):

    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)

    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    model = model.to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    data_per_rank = data[rank::world_size]

    # Process the data in parallel
    all_preds = []
    for entry in tqdm(data_per_rank, desc=f"Rank {rank} processing"):
        with torch.no_grad():
            inputs = tokenizer(entry['llm_prompt'], return_tensors='pt').to(device)
            cut_at = inputs.input_ids.shape[1]

            if args.data_prefix not in ['default', 'default_cfc']:
                structure_ids = entry['structure_ids'].to(device)
                num_structure_tokens = entry['num_structure_tokens'].to(device)
                cur_pred = model.module.generate(  # Use model.module to access the original model inside DDP
                    **inputs,
                    do_sample=args.do_sample,
                    structure_values=structure_ids,
                    num_structure_tokens=num_structure_tokens,
                    max_new_tokens=args.gen_length)
            else:
                cur_pred = model.module.generate(
                    **inputs,
                    do_sample=args.do_sample,
                    max_new_tokens=args.gen_length)

            prediction = tokenizer.decode(cur_pred[0][cut_at:], skip_special_tokens=True)

            # Manual removal of special tokens
            # if 'qwen' in args.text_model_id.lower():
            #     prediction = re.sub(PATTERN, "", prediction)

            all_preds.append({
                "task_id": entry["metadata"]["task_id"],
                "pred": truncate_trash(prediction),
            })

    # Collect results from all GPUs (using all_gather)
    local_preds = all_preds
    all_preds = [None for _ in range(world_size)]
    dist.all_gather_object(all_preds, local_preds)

    # Save results to disk on rank 0
    if rank == 0:
        with open(f"{args.output_dir}/prediction.jsonl", "w", encoding="utf-8") as f_pred:
            for entry in all_preds:
                if isinstance(entry, list):
                    for entry_ in entry:
                        f_pred.write(json.dumps(entry_) + "\n")
                else:
                    f_pred.write(json.dumps(entry_) + "\n")

    dist.destroy_process_group()


if __name__ == "__main__":

    args = parse_args()
    print('Input args:', args)

    code_tokenizer = AutoTokenizer.from_pretrained(args.text_model_id, use_fast=False)
    code_tokenizer.add_tokens([STRUCTURE_TOKEN])
    if code_tokenizer.pad_token_id is None:
        code_tokenizer.pad_token_id = code_tokenizer.eos_token_id
    structure_token_id = code_tokenizer.convert_tokens_to_ids(STRUCTURE_TOKEN)

    structure_tokenizer = AutoTokenizer.from_pretrained(args.structure_model_id, use_fast=False)

    structure_config = AutoConfig.from_pretrained(args.structure_model_id)
    structure_config.model_id = args.structure_model_id
    structure_config.pad_token_id = structure_tokenizer.pad_token_id

    text_config = AutoConfig.from_pretrained(args.text_model_id)
    text_config.model_id = args.text_model_id
    assert len(code_tokenizer) <= text_config.vocab_size, 'The tokenizer length is larger than the embedding layer shape, resize the embeddings'
    configuration = LlavaCodeConfig(structure_config, text_config,
                                    pad_token_id=code_tokenizer.pad_token_id,
                                    structure_token_id=structure_token_id,
                                    injector=False)

    if args.model_checkpoint is not None:
        logger.info(f"Loading checkpoint: {args.model_checkpoint}")
        model = LlavaCodeForConditionalGeneration.load_from_checkpoint(
            args.model_checkpoint, config=configuration)
    else:
        model = LlavaCodeForConditionalGeneration(configuration)
    if args.projector_checkpoint:
        print(f'Loading projection weighs from {args.projector_checkpoint}')
        model.multi_modal_projector.load_state_dict(torch.load(args.projector_checkpoint))
    model.eval()

    fim_tokens = get_fim_tokens(args.text_model_id)
    print('fim tokens:', fim_tokens)
    data = build_dataset(args, code_tokenizer, structure_tokenizer, fim_tokens)

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'

    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(main_worker, nprocs=world_size, args=(world_size, model, code_tokenizer, data, args))

    if args.compute_cceval_metric:
        compute_metric_stmt_cceval(args)
    else:
        compute_metric_stmt(args)
