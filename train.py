import os
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

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
from lightning.pytorch.utilities import rank_zero_only

from huggingface_hub import login
from dotenv import load_dotenv
import logging

from models import LlavaCodeConfig,  LlavaCodeForConditionalGeneration
from pl_args import add_model_args, add_pl_args, add_program_args
from datamodule import LlavaCodeDataModule
from datamodule.const import STRUCTURE_TOKEN, FIMMAP
from datamodule.utils import get_fim_tokens
from pl_logger import ClearMLLogger

load_dotenv()
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
    if len(code_tokenizer) > text_config.vocab_size:
        print(f'Resizing model embeddings to a new vocab size of {text_config.vocab_size + 1}')
        text_config.vocab_size = text_config.vocab_size + 1  # for a new <CODE_STRUCTURE>
    configuration = LlavaCodeConfig(structure_config, text_config,
                                    pad_token_id=code_tokenizer.pad_token_id,
                                    structure_token_id=structure_token_id,
                                    quantize=args.quantize,
                                    projector=args.projector)

    if args.model_checkpoint is not None:
        logger.info(f"Loading checkpoint: {args.model_checkpoint}")
        model = LlavaCodeForConditionalGeneration.load_from_checkpoint(
            args.model_checkpoint, config=configuration)
        # model = LlavaCodeForConditionalGeneration(configuration)
        # ckpt = torch.load(args.model_checkpoint, map_location="cpu")
        # state_dict = ckpt["state_dict"]
        # projector_state_dict = {
        #     k.replace("model.multi_modal_projector.", ""): v
        #     for k, v in state_dict.items()
        #     if k.startswith("model.multi_modal_projector.")}
        # model.multi_modal_projector.load_state_dict(projector_state_dict)

        projector_state_dict = model.multi_modal_projector.state_dict()
        save_name = f"{os.path.basename(args.model_checkpoint).removesuffix('.ckpt')}_projector.pth"
        save_dir = os.path.dirname(args.model_checkpoint)
        torch.save(projector_state_dict, os.path.join(save_dir, save_name))
        import sys
        sys.exit(0)
    else:
        model = LlavaCodeForConditionalGeneration(configuration)
    if args.projector_checkpoint:
        print(f'Loading projection weighs from: {args.projector_checkpoint}')
        model.multi_modal_projector.load_state_dict(torch.load(args.projector_checkpoint), strict=True)

    for p in model.model.structure_model.parameters():
        p.requires_grad = False

    if args.training_stage == 0 or args.training_stage == 1 or args.training_stage == 2:
        for p in model.model.language_model.parameters():
            p.requires_grad = False
        for p in model.lm_head.parameters():
            p.requires_grad = False

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
        fim_tokens=get_fim_tokens(args.text_model_id),
        training_stage=args.training_stage,
        num_workers=args.num_workers,
        code_tokenizer=code_tokenizer,
        structure_tokenizer=structure_tokenizer,
        structure_token_id=structure_token_id,
        num_structure_tokens=args.num_structure_tokens,
        language=args.language
    )
    print('Training stage:', args.training_stage)
    import datasets
    print('datasetsver', datasets.__version__)
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

    tags = [args.text_model_id.split('/')[-1], args.structure_model_id.split('/')[-1], 'TheStack2', args.language]
    clearml_logger = ClearMLLogger(args.log_dir, project_name='LlavaCode', task_name=args.exp_name, tags=tags)
    csv_logger = CSVLogger(args.log_dir, name=args.exp_name, version="")

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
    logger.info('Initialized trainer')

    trainer.validate(model, datamodule=data)
    logger.info('Finished validation')

    trainer.fit(model, data)
    logger.info('Finished training')

    rank_zero_only(trainer.logger._task.close())
