from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    RobertaConfig,
)

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.strategies.ddp import DDPStrategy

from huggingface_hub import login
from dotenv import load_dotenv
import logging
from clearml import Task, Logger

from parser import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token,
                   tree_to_variable_index)
from tree_sitter import Language, Parser
from preprocess import AST

from modeling_llava_code import LlavaCodeConfig,  LlavaCodeForConditionalGeneration
from pl_args import add_model_args, add_pl_args, add_program_args
from pl_data import DataModule

load_dotenv()
import os
token = os.getenv("HF_TOKEN")
login(token=token)

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


if __name__ == "__main__":

    parser = add_program_args()
    parser = add_model_args(parser)
    parser = add_pl_args(parser)
    args = parser.parse_args()
    seed_everything(args.seed, workers=True)

    # User gives batch size over all GPUs, PL requires per GPU
    args.train_batch_size = args.train_batch_size // (args.devices * args.num_nodes)
    args.valid_batch_size = args.valid_batch_size // (args.devices * args.num_nodes)
    logger.info(f'{args.train_batch_size=} {args.valid_batch_size=}')
    # User gives validation check interval in terms of number of steps, PL requires in terms of batches
    args.val_check_interval *= args.accumulate_grad_batches

    structure_config = RobertaConfig.from_pretrained(args.structure_model_id)
    structure_config.model_id = args.structure_model_id
    text_config = AutoConfig.from_pretrained(args.text_model_id)
    text_config.model_id = args.text_model_id
    text_config.vocab_size = text_config.vocab_size + 1 # for a new <CODE_STRUCTURE>
    configuration = LlavaCodeConfig(structure_config, text_config, structure_token_id=49152)

    model = LlavaCodeForConditionalGeneration(configuration)
    code_tokenizer = AutoTokenizer.from_pretrained(args.text_model_id)

    # freezing everything but the multi_model_projector parameters
    for p in model.model.structure_model.parameters():
        p.requires_grad = False
    for p in model.model.language_model.parameters():
        p.requires_grad = False
    trainable_params, all_params = 0, 0
    for name, param in model.named_parameters():
        all_params += param.numel()
        trainable_params += param.numel() * param.requires_grad
    logger.info(f'Trainable parameters: {trainable_params}, All Parameters: {all_params}, Percentage: {trainable_params / all_params * 100 :.2f}%')

    data = DataModule(
        args.data_prefix,
        args.train_datadir, 
        args.valid_datadir, 
        args.train_batch_size, 
        args.valid_batch_size,
        num_workers=args.num_workers,
        code_tokenizer=code_tokenizer,
        ast_tokenizer=model.model.structure_model.tokenizer
    )
    data.setup()
    args.num_training_examples = len(data.train_dataloader())

    logger.info('Initializing PL Trainer...')
    custom_trainer_kwargs = {
        # 'callbacks': callbacks,
        # 'logger': loggers,
        # 'strategy': DeepSpeedStrategy(config=args.ds_config) \
        #     if args.use_deepspeed else DDPStrategy(find_unused_parameters=False),
        'num_nodes': args.num_nodes,
        # 'plugins': plugins,
        'precision': args.precision,
        'accelerator':args.accelerator,
        'devices':args.devices,
        'max_epochs':args.max_epochs,
        'max_steps':args.max_steps,
        'val_check_interval':args.val_check_interval,
        'log_every_n_steps':args.log_every_n_steps,
        'accumulate_grad_batches':args.accumulate_grad_batches,
        'gradient_clip_val':args.gradient_clip_val,
        'default_root_dir':args.default_root_dir,
    }
    trainer = pl.Trainer(**custom_trainer_kwargs)
    logger.warning(f'{trainer.__dict__=}')

    model.set_trainer_args(args)

    trainer.fit(model, data)

    logger.info('Finished trainingvim')
