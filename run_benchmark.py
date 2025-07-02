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


def prepare_prompt(tokenizer, 
                   ast_tokenizer, 
                   task, 
                   left_cxt, 
                   right_cxt=None, 
                   max_ast_length=512, 
                   use_code_structure=False):
    if use_code_structure:
        ast_tokens = AST(left_cxt, 'python', ast_tokenizer)
        patch_length = max_ast_length - 4 # 4 special tokens for unixcoder
        num_structure_tokens = math.ceil(len(ast_tokens) / patch_length)
        
        ast_ids = []
        for i in range(num_structure_tokens):
            patch = ast_tokens[i * patch_length : (i + 1) * patch_length]
            patch_tokens = [ast_tokenizer.cls_token, "<encoder-only>", ast_tokenizer.sep_token] + patch + [ast_tokenizer.sep_token]
            patch_ids = ast_tokenizer.convert_tokens_to_ids(patch_tokens)
            ast_ids.extend(patch_ids)
        ast_ids = torch.tensor(ast_ids, dtype=torch.long)
        ast_length = len(ast_ids)
        ast_ids = F.pad(ast_ids, (0, max_ast_length * num_structure_tokens - ast_length), value=ast_tokenizer.pad_token_id)
        
        left_cxt_truncated = tokenizer.decode(tokenizer.encode(left_cxt)[-(args.max_seq_length - args.gen_length - num_structure_tokens - args.right_context_length):])
        right_cxt_truncated = tokenizer.decode(tokenizer.encode(right_cxt)[:args.right_context_length])
        prompt = f'<fim_prefix>{left_cxt_truncated}' + '<CODE_STRUCTURE>' * num_structure_tokens + f'<fim_suffix>{right_cxt_truncated}<fim_middle>'

        return prompt, ast_ids, torch.tensor([num_structure_tokens])
        
    else:
        
        left_cxt_truncated = tokenizer.decode(tokenizer.encode(left_cxt)[-(args.max_seq_length - args.gen_length - args.right_context_length):])
        right_cxt_truncated = tokenizer.decode(tokenizer.encode(right_cxt)[:args.right_context_length])
        prompt = f'<fim_prefix>{left_cxt_truncated}' + f'<fim_suffix>{right_cxt_truncated}<fim_middle>'

        return prompt, None, None


def build_dataset(args, code_tokenizer, ast_tokenizer):
    with open(args.prompt_file) as f:
        raw_data = [json.loads(line) for line in f.readlines()]

    data = []
    for entry in raw_data:
        task = args.task

        left_cxt = entry["prompt"]
        right_cxt = entry["right_context"]
        entry['llm_prompt'], entry['ast_ids'], entry['num_structure_tokens'] = \
            prepare_prompt(code_tokenizer, ast_tokenizer, task, 
                           left_cxt, right_cxt, args.max_ast_length,
                           args.use_code_structure)

        data.append(entry)

    return data


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--language", type=str, required=True, help="language name")
    parser.add_argument("--model_checkpoint", type=str)

    parser.add_argument("--prompt_file", type=str, default=None, help="file with a list of prompts")
    parser.add_argument("--gen_length", type=int, default=50, help="max length of generated token sequence")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="max length of prompt")
    parser.add_argument("--max_ast_length", type=int, default=512, help="max length of ast sequence")
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

    parser.add_argument("--use_code_structure", action="store_true")

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
    text_config.vocab_size = text_config.vocab_size + 1 # for a new <CODE_STRUCTURE>
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
            if args.use_code_structure:
                
                ast_ids = entry['ast_ids'].to(device)
                num_structure_tokens = entry['num_structure_tokens'].to(device)
                cur_pred = model.generate(**inputs,
                                          use_cache=True,
                                          structure_values=ast_ids,
                                          num_structure_tokens=num_structure_tokens,
                                          max_new_tokens=args.gen_length,
                                          bad_words_ids=[[configuration.structure_token_id]])
            
            else:
                
                cur_pred = model.generate(**inputs,
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
