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
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

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


class CheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
        self,
        save_step_frequency=5000,
        prefix="NStep-ckpt",
        use_modelcheckpoint_filename=False,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

    def on_batch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if (global_step > 0) and global_step % self.save_step_frequency == 0:
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"{self.prefix}_{epoch=}_{global_step=}.ckpt"
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)


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

    code_tokenizer = AutoTokenizer.from_pretrained(args.text_model_id, use_fast=False)
    code_tokenizer.add_tokens(['<CODE_STRUCTURE>'])
    if code_tokenizer.pad_token_id is None:
        code_tokenizer.pad_token_id = code_tokenizer.eos_token_id

    structure_config = RobertaConfig.from_pretrained(args.structure_model_id)
    structure_config.model_id = args.structure_model_id
    text_config = AutoConfig.from_pretrained(args.text_model_id)
    text_config.model_id = args.text_model_id
    text_config.vocab_size = text_config.vocab_size + 1 # for a new <CODE_STRUCTURE>
    configuration = LlavaCodeConfig(structure_config, text_config,
                                    pad_token_id=code_tokenizer.pad_token_id,
                                    structure_token_id=49152)

    if args.resume_from_checkpoint:
        model = LlavaCodeForConditionalGeneration.load_from_checkpoint(args.checkpoint_path, config=configuration)
    else:
        model = LlavaCodeForConditionalGeneration(configuration)

    # Stage 1: only projection is trained
    # Stage 2: projection and llm are trained
    # structure model weights are always frozen
    for p in model.model.structure_model.parameters():
        p.requires_grad = False

    if args.training_stage == 1:
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
        ast_tokenizer=model.model.structure_model.tokenizer,
        structure_token_id=49152
    )
    data.setup()
    args.num_training_examples = len(data.train_dataloader())

    callbacks = []
    callbacks = [LearningRateMonitor(logging_interval='step')]
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="Valid/Loss/MLE",
        mode="min",
        every_n_train_steps=args.save_step_frequency
    )
    callbacks.append(checkpoint_callback)
    callbacks.append(CheckpointEveryNSteps(save_step_frequency=args.save_step_frequency))

    logger.info('Initializing PL Trainer...')
    custom_trainer_kwargs = {
        'callbacks': callbacks,
        # 'logger': loggers,
        'strategy': DeepSpeedStrategy(config=args.ds_config) \
            if args.use_deepspeed else DDPStrategy(find_unused_parameters=False),
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
        'gradient_clip_algorithm': 'norm',
        'default_root_dir':args.default_root_dir,
    }
    trainer = pl.Trainer(**custom_trainer_kwargs)
    logger.warning(f'{trainer.__dict__=}')

    model.set_trainer_args(args)

    trainer.fit(model, data)

    logger.info('Finished training')
