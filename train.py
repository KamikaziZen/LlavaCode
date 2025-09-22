import os

from transformers import (
    AutoConfig,
    AutoTokenizer,
    RobertaConfig,
    RobertaTokenizer,
)

import torch
import lightning.pytorch as pl
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from huggingface_hub import login
from dotenv import load_dotenv
import logging

from models import LlavaCodeConfig,  LlavaCodeForConditionalGeneration
from pl_args import add_model_args, add_pl_args, add_program_args
from datamodule import LlavaCodeDataModule
from datamodule.const import STRUCTURE_TOKEN, FIMMAP
from pl_logger import ClearMLLogger

load_dotenv()
token = os.getenv("HF_TOKEN")
login(token=token)

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# class CheckpointEveryNSteps(pl.Callback):
#     """
#     Save a checkpoint every N steps, instead of Lightning's default that checkpoints
#     based on validation loss.
#     """

#     def __init__(
#         self,
#         save_step_frequency=5000,
#         prefix="NStep-ckpt",
#         use_modelcheckpoint_filename=False,
#     ):
#         """
#         Args:
#             save_step_frequency: how often to save in steps
#             prefix: add a prefix to the name, only used if
#                 use_modelcheckpoint_filename=False
#             use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
#                 default filename, don't use ours.
#         """
#         self.save_step_frequency = save_step_frequency
#         self.prefix = prefix
#         self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

#     def on_batch_end(self, trainer: pl.Trainer, _):
#         """ Check if we should save a checkpoint after every train batch """
#         epoch = trainer.current_epoch
#         global_step = trainer.global_step
#         if (global_step > 0) and global_step % self.save_step_frequency == 0:
#             if self.use_modelcheckpoint_filename:
#                 filename = trainer.checkpoint_callback.filename
#             else:
#                 filename = f"{self.prefix}_{epoch=}_{global_step=}.ckpt"
#             ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
#             trainer.save_checkpoint(ckpt_path)


if __name__ == "__main__":

    parser = add_program_args()
    parser = add_model_args(parser)
    parser = add_pl_args(parser)
    args = parser.parse_args()
    pl.seed_everything(args.seed, workers=True)

    # User gives batch size over all GPUs, PL requires per GPU
    args.train_batch_size = args.train_batch_size // (args.devices * args.num_nodes)
    args.valid_batch_size = args.valid_batch_size // (args.devices * args.num_nodes)
    logger.info(f'{args.train_batch_size=} {args.valid_batch_size=}')
    # User gives validation check interval in terms of number of steps, PL requires in terms of batches
    args.val_check_interval *= args.accumulate_grad_batches

    print('args:', args)

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
    text_config.vocab_size = text_config.vocab_size + 1  # for a new <CODE_STRUCTURE>
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
        print('Loading projection weighs')
        model.multi_modal_projector.load_state_dict(torch.load(args.projector_checkpoint))

    # Stage 0: only projection is trained on entropy loss
    # Stage 1: only projection is trained on entropy loss and KL loss
    # Stage 2: projection and llm are trained on entropy loss
    # structure model weights are always frozen
    for p in model.model.structure_model.parameters():
        p.requires_grad = False

    if args.training_stage == 0 or args.training_stage == 1 or args.training_stage == 2:
        for p in model.model.language_model.parameters():
            p.requires_grad = False

    # unfreezing Q and V of the first attention block
    # if args.training_stage == 0 or args.training_stage == 1:
    #     for name, p in model.named_parameters():
    #         if name.startswith('model.structure_model.model.encoder.layer.0.attention.self.query'):
    #             p.requires_grad = True
    #         if name.startswith('model.structure_model.model.encoder.layer.0.attention.self.value'):
    #             p.requires_grad = True

    trainable_params, all_params = 0, 0
    for name, param in model.named_parameters():
        all_params += param.numel()
        trainable_params += param.numel() * param.requires_grad

    print(f"""Trainable parameters: {trainable_params},
              All Parameters: {all_params},
              Percentage: {trainable_params / all_params * 100 :.2f}%""")

    total_norm = 0.0
    for p in model.model.multi_modal_projector.parameters():
        total_norm += p.data.norm(2).item() ** 2
    print(f"Total norm of projector weights: {total_norm}")

    data = LlavaCodeDataModule(
        args.data_prefix,
        args.train_datadir,
        args.valid_datadir,
        args.train_batch_size,
        args.valid_batch_size,
        fim_tokens=model.model.fim_tokens,
        training_stage=args.training_stage,
        num_workers=args.num_workers,
        code_tokenizer=code_tokenizer,
        structure_tokenizer=structure_tokenizer,
        structure_token_id=structure_token_id,
        num_structure_tokens=args.num_structure_tokens,
    )
    print('Training stage:', args.training_stage)
    data.setup()
    args.num_training_examples = len(data.train_dataloader())

    callbacks = []
    callbacks = [LearningRateMonitor(logging_interval='step')]
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="Val_Acc_EM",
        save_last=False,
        mode="max",
        filename="{epoch}-{step}-{Val_Acc_EM:.4f}-{Val_Acc_ES:.4f}",
    )
    callbacks.append(checkpoint_callback)

    tags = [args.text_model_id.split('/')[-1], args.structure_model_id.split('/')[-1]]
    clearml_logger = ClearMLLogger(project_name='LlavaCode', task_name=args.exp_name, tags=tags)
    csv_logger = CSVLogger("lightning_logs/", name=args.exp_name, version="")

    logger.info('Initializing PL Trainer...')
    custom_trainer_kwargs = {
        "num_sanity_val_steps": 0,
        'callbacks': callbacks,
        'logger': [clearml_logger, csv_logger],
        'strategy': DeepSpeedStrategy(config=args.ds_config) \
            if args.use_deepspeed else DDPStrategy(find_unused_parameters=False),
        'num_nodes': args.num_nodes,
        # 'plugins': plugins,
        'precision': args.precision,
        'accelerator': args.accelerator,
        'devices': args.devices,
        'max_epochs': args.max_epochs,
        'max_steps': args.max_steps,
        'val_check_interval': args.val_check_interval,
        'log_every_n_steps': args.log_every_n_steps,
        'accumulate_grad_batches': args.accumulate_grad_batches,
        'gradient_clip_val': args.gradient_clip_val,
        'gradient_clip_algorithm': 'norm',
        'default_root_dir': args.default_root_dir,
        # 'limit_val_batches': 0.0
    }

    trainer = pl.Trainer(**custom_trainer_kwargs)
    logger.warning(f'{trainer.__dict__=}')

    model.set_trainer_args(args)

    trainer.validate(model, datamodule=data)

    trainer.fit(model, data)

    save_path = os.path.join(f'lightning_logs/{args.exp_name}', f"projector_weights_{args.max_epochs}ep.pt")
    torch.save(model.multi_modal_projector.state_dict(), save_path)
    logger.info(f"Saved weights to {save_path}")

    trainer.logger._task.close()
    logger.info('Finished training')