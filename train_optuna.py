import os
import math
import logging
from typing import Dict, Any
from types import SimpleNamespace

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DDPStrategy, DeepSpeedStrategy

from transformers import AutoConfig, AutoTokenizer
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("HF_TOKEN")
if token:
    try:
        login(token=token)  # set HF_TOKEN in env or .env to avoid entering code each run
    except Exception as e:
        log = logging.getLogger(__name__)
        log.warning(f"Hugging Face login failed: {e}")
else:
    log = logging.getLogger(__name__)
    log.info("HF_TOKEN not set; skipping HF login (models must be public or env var must be set).")

from omegaconf import DictConfig, OmegaConf
import hydra

from models import LlavaCodeConfig, LlavaCodeForConditionalGeneration
from datamodule import LlavaCodeDataModule
from datamodule.const import STRUCTURE_TOKEN
from pl_logger import ClearMLLogger 


log = logging.getLogger(__name__)


# class CheckpointEveryNSteps(pl.Callback):
#     """Save a checkpoint every N *training* steps."""
#     def __init__(self, save_step_frequency: int = 5000, prefix: str = "NStep-ckpt", use_modelcheckpoint_filename: bool = False):
#         self.save_step_frequency = int(save_step_frequency)
#         self.prefix = prefix
#         self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

#     def on_train_batch_end(self, trainer: pl.Trainer, *_):  # PL>=2.0 hook
#         global_step = trainer.global_step
#         if global_step > 0 and global_step % self.save_step_frequency == 0:
#             if self.use_modelcheckpoint_filename:
#                 filename = trainer.checkpoint_callback.filename
#             else:
#                 filename = f"{self.prefix}_epoch={trainer.current_epoch}_global_step={global_step}.ckpt"
#             ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
#             trainer.save_checkpoint(ckpt_path)



def _build_model(cfg: DictConfig):

    code_tok = AutoTokenizer.from_pretrained(cfg.model.text_model_id, use_fast=False)
    code_tok.add_tokens([STRUCTURE_TOKEN])
    if code_tok.pad_token_id is None:
        code_tok.pad_token_id = code_tok.eos_token_id
    structure_token_id = code_tok.convert_tokens_to_ids(STRUCTURE_TOKEN)

    structure_tok = AutoTokenizer.from_pretrained(cfg.model.structure_model_id, use_fast=False)

    structure_conf = AutoConfig.from_pretrained(cfg.model.structure_model_id)
    structure_conf.model_id = cfg.model.structure_model_id
    structure_conf.pad_token_id = structure_tok.pad_token_id

    text_conf = AutoConfig.from_pretrained(cfg.model.text_model_id)
    text_conf.model_id = cfg.model.text_model_id
    text_conf.vocab_size = text_conf.vocab_size + 1  # for the new <CODE_STRUCTURE>

    llava_conf = LlavaCodeConfig(
        structure_conf,
        text_conf,
        pad_token_id=code_tok.pad_token_id,
        structure_token_id=structure_token_id,
        injector=False,
    )

    if cfg.trainer.model_checkpoint:
        log.info(f"Loading model from checkpoint: {cfg.trainer.model_checkpoint}")
        model = LlavaCodeForConditionalGeneration.load_from_checkpoint(cfg.trainer.model_checkpoint, config=llava_conf)
    else:
        model = LlavaCodeForConditionalGeneration(llava_conf)

    if cfg.trainer.projector_checkpoint:
        log.info("Loading projector weights")
        state = torch.load(cfg.trainer.projector_checkpoint, map_location="cpu")
        model.multi_modal_projector.load_state_dict(state)

    # Freeze/unfreeze per training_stage
    for p in model.model.structure_model.parameters():
        p.requires_grad = False

    if cfg.trainer.training_stage in (0, 1, 2):
        for p in model.model.language_model.parameters():
            p.requires_grad = False

    trainable, total = 0, 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    log.info(f"Trainable parameters: {trainable:,} / {total:,} ({100.0*trainable/max(total,1):.2f}%)")

    try:
        total_norm_sq = sum((p.data.norm(2).item() ** 2) for p in model.model.multi_modal_projector.parameters())
        log.info(f"Projector L2 norm^2: {total_norm_sq:.4f}")
    except Exception:
        pass

    return model, code_tok, structure_tok, structure_token_id


def _build_datamodule(cfg: DictConfig, code_tok, structure_tok, structure_token_id, fim_tokens):
    # Convert *global* batch sizes to per-device ones
    # devices = int(cfg.trainer.devices)
    # nodes = int(cfg.trainer.num_nodes)
    # world = max(1, devices * nodes)

    # train_bsz_total = int(cfg.data.train_batch_size)
    # valid_bsz_total = int(cfg.data.valid_batch_size)

    # per_device_train = max(1, train_bsz_total // world)
    # per_device_valid = max(1, valid_bsz_total // world)

    # # Convert val_check_interval in update steps (user perspective) -> PL batches (effective with accumulation)
    # val_check_batches = int(cfg.trainer.val_check_interval) * max(1, int(cfg.trainer.accumulate_grad_batches))

    # # Write back so the model/trainer can see consistent values
    # cfg.data.train_batch_size = per_device_train
    # cfg.data.valid_batch_size = per_device_valid
    # cfg.trainer.val_check_interval = val_check_batches

    devices = int(cfg.trainer.devices)
    nodes = int(cfg.trainer.num_nodes)
    world = max(1, devices * nodes)

    train_bsz_total = int(cfg.data.train_batch_size)
    valid_bsz_total = int(cfg.data.valid_batch_size)

    per_device_train = max(1, train_bsz_total // world)
    per_device_valid = max(1, valid_bsz_total // world)

    # Validation frequency
    vci = cfg.trainer.val_check_interval
    if isinstance(vci, float) and 0 < vci <= 1:
        # fraction of an epoch: pass through unchanged (Trainer interprets it)
        val_check_interval = vci
    else:
        # integer update-steps
        vci_int = int(vci)
        agb = max(1, int(cfg.trainer.accumulate_grad_batches))
        val_check_interval = vci_int * agb

    cfg.data.train_batch_size = per_device_train
    cfg.data.valid_batch_size = per_device_valid
    cfg.trainer.val_check_interval = val_check_interval

    dm = LlavaCodeDataModule(
        cfg.data.data_prefix,
        cfg.data.train_datadir,
        cfg.data.valid_datadir,
        per_device_train,
        per_device_valid,
        fim_tokens=fim_tokens,
        training_stage=cfg.trainer.training_stage,
        num_workers=cfg.data.num_workers,
        code_tokenizer=code_tok,
        structure_tokenizer=structure_tok,
        structure_token_id=structure_token_id,
        num_structure_tokens=cfg.model.num_structure_tokens,
    )
    return dm


def _build_trainer(cfg: DictConfig, clearml_logger, csv_logger) -> pl.Trainer:
    callbacks = [LearningRateMonitor(logging_interval='step')]

    if cfg.logging.save_top_k > 0:
        ckpt_cb = ModelCheckpoint(
            save_top_k=cfg.logging.save_top_k,
            monitor=cfg.optuna.monitor,
            mode=cfg.optuna.mode,
            save_last=False,
            # every_n_train_steps=cfg.trainer.save_step_frequency,
            filename="epoch={epoch}-step={step}-metric={" + cfg.optuna.monitor.replace("/", "_") + ":.4f}",
            auto_insert_metric_name=False,
        )
        callbacks.append(ckpt_cb)

  # Early stopping on the monitored validation metricб patience counts validation checks
    if getattr(cfg, "early_stopping", None) and cfg.early_stopping.enabled:
        es_cb = EarlyStopping(
            monitor=cfg.optuna.monitor,
            mode=cfg.optuna.mode,
            patience=int(cfg.early_stopping.patience),
            min_delta=float(cfg.early_stopping.min_delta),
            verbose=True,
        )
        callbacks.append(es_cb)

    # if cfg.logging.checkpoint_every_n_steps > 0:
    #     callbacks.append(CheckpointEveryNSteps(cfg.logging.checkpoint_every_n_steps))

    strategy = (
        DeepSpeedStrategy(config=cfg.trainer.ds_config)
        if cfg.trainer.use_deepspeed
        else DDPStrategy(find_unused_parameters=False)
    )


    # debug knobs
    extra = {}
    for k in [
        'fast_dev_run',            
        'limit_train_batches',    
        'limit_val_batches',       
        'limit_test_batches',     
        'overfit_batches',        
    ]:
        v = getattr(cfg.trainer, k, None)
        if v is not None:
            extra[k] = v

    trainer = pl.Trainer(
        num_sanity_val_steps=0,
        callbacks=callbacks,
        logger=[clearml_logger, csv_logger],
        strategy=strategy,
        num_nodes=cfg.trainer.num_nodes,
        precision=cfg.trainer.precision,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        max_epochs=cfg.trainer.max_epochs,
        max_steps=cfg.trainer.max_steps,
        val_check_interval=cfg.trainer.val_check_interval,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        gradient_clip_algorithm='norm',
        default_root_dir=cfg.logging.default_root_dir,
        **extra,
    )

    return trainer


def _get_objective_from_metrics(metrics: Dict[str, Any], key: str) -> float:
    if key in metrics:
        return float(metrics[key])

    for k, v in metrics.items():
        if isinstance(k, str) and (k.endswith(key) or key in k):
            try:
                return float(v)
            except Exception:
                continue

    for v in metrics.values():
        try:
            return float(v)
        except Exception:
            pass
    raise RuntimeError(f"Objective metric '{key}' not found in metrics: {list(metrics.keys())}")


def _make_args_compat(cfg: DictConfig) -> SimpleNamespace:
    """Build an argparse-like flat namespace so existing model code can do args.foo.
    Maps nested Hydra keys to the original argparse names.
    """
    d = dict(
        # model
        text_model_id=cfg.model.text_model_id,
        structure_model_id=cfg.model.structure_model_id,
        num_structure_tokens=cfg.model.num_structure_tokens,
        loss=cfg.model.loss,
        dropout_layers=cfg.model.dropout_layers,
        dropout_p=cfg.model.dropout_p,
        lr_scheduler_type=cfg.model.lr_scheduler_type,
        full_sequence_code_completion_loss=cfg.model.full_sequence_code_completion_loss,
        functional_dropout=cfg.model.functional_dropout,
        debug_disable_adding_new_token=cfg.model.debug_disable_adding_new_token,
        # data
        data_prefix=cfg.data.data_prefix,
        train_datadir=cfg.data.train_datadir,
        valid_datadir=cfg.data.valid_datadir,
        train_batch_size=cfg.data.train_batch_size,
        valid_batch_size=cfg.data.valid_batch_size,
        num_workers=cfg.data.num_workers,
        # trainer/runtime
        val_check_interval=cfg.trainer.val_check_interval,
        devices=cfg.trainer.devices,
        num_nodes=cfg.trainer.num_nodes,
        accelerator=cfg.trainer.accelerator,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        num_training_examples=getattr(cfg.trainer, 'num_training_examples', -1),
        max_steps=cfg.trainer.max_steps,
        max_epochs=cfg.trainer.max_epochs,
        default_root_dir=cfg.logging.default_root_dir,
        use_deepspeed=cfg.trainer.use_deepspeed,
        precision=cfg.trainer.precision,
        ds_config=cfg.trainer.ds_config,
        model_checkpoint=cfg.trainer.model_checkpoint,
        projector_checkpoint=cfg.trainer.projector_checkpoint,
        seed=cfg.trainer.seed,
        training_stage=cfg.trainer.training_stage,
        # save_step_frequency=cfg.trainer.save_step_frequency,
        debug_cuda_mem=cfg.trainer.debug_cuda_mem,
        warmup_steps=cfg.trainer.warmup_steps,
        weight_decay=cfg.trainer.weight_decay,
        kl_temperature=cfg.trainer.kl_temperature,
        distill_topk=getattr(cfg.trainer, 'distill_topk', None),
        # logging
        exp_name=cfg.logging.exp_name,
        log_dir=cfg.logging.default_root_dir,
        # training hparams
        lr=cfg.training.lr,
        alpha_ce=cfg.training.alpha_ce,
        alpha_align=cfg.training.alpha_align,
        alpha_scst=cfg.training.alpha_scst,
        alpha_kl=cfg.training.alpha_kl,
    )
    return SimpleNamespace(**d)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> float:
    pl.seed_everything(int(cfg.trainer.seed), workers=True)
    log.setLevel(logging.INFO)
    log.info(OmegaConf.to_yaml(cfg, resolve=True))

    model, code_tok, structure_tok, structure_token_id = _build_model(cfg)
    data = _build_datamodule(cfg, code_tok, structure_tok, structure_token_id, getattr(model.model, 'fim_tokens', None))

    # Loggers
    tags = [cfg.model.text_model_id.split('/')[-1], cfg.model.structure_model_id.split('/')[-1]]
    cfg.logging.exp_name += f"_{cfg.training.lr}lr_{cfg.trainer.devices * trainer.accumulate_grad_batches}b"
    cfg.logging.exp_name += f"_ce{cfg.training.alpha_ce}_al{training.alpha_align}_scst{training.alpha_scst}_kl{training.alpha_kl}"
    clearml_logger = ClearMLLogger(project_name=cfg.clearml_project_folder, task_name=cfg.logging.exp_name, tags=tags)
    csv_logger = CSVLogger(save_dir="lightning_logs/", name=cfg.logging.exp_name, version="")

    trainer = _build_trainer(cfg, clearml_logger, csv_logger)
    clearml_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    if hasattr(model, "set_trainer_args"):
        model.set_trainer_args(_make_args_compat(cfg))

    # _ = trainer.validate(model, datamodule=data)

    trainer.fit(model, datamodule=data)

    # Save projector weights
    save_path = os.path.join(trainer.default_root_dir or os.getcwd(), f"projector_weights_{cfg.trainer.max_epochs}ep.pt")
    try:
        save_dir = trainer.default_root_dir or os.getcwd()
        os.makedirs(save_dir, exist_ok=True)  #создаём, если  нет
        save_path = os.path.join(save_dir, f"projector_weights_{cfg.trainer.max_epochs}ep.pt")
        torch.save(model.multi_modal_projector.state_dict(), save_path)
    except Exception as e:
        log.warning(f"Failed to save projector weights: {e}")


    objective = None
    from lightning.pytorch.callbacks import ModelCheckpoint as _MC
    for cb in trainer.callbacks:
        if isinstance(cb, _MC) and getattr(cb, "monitor", None) == cfg.optuna.monitor:
            if getattr(cb, "best_model_score", None) is not None:
                objective = float(cb.best_model_score.item())
                log.info(f"Optuna objective (BEST): {cfg.optuna.monitor}={objective}")
            else:
                log.warning("ModelCheckpoint has no best_model_score; did validation log the monitored metric?")
            break

    if objective is None:
        val_results = trainer.validate(model, datamodule=data)
        metrics = val_results[0] if isinstance(val_results, list) and val_results else {}
        objective = _get_objective_from_metrics(metrics, cfg.optuna.monitor)
        log.info(f"Optuna objective (FINAL val fallback): {cfg.optuna.monitor}={objective}")

    try:
        trainer.logger._task.close()
    except Exception:
        pass
    return float(objective)


if __name__ == "__main__":
    main()
