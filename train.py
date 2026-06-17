#!/usr/bin/env python3
import argparse
import os
import sys
import time
import datetime
import logging
from pathlib import Path
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import yaml
import wandb

def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes:.0f}m{secs:.0f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours:.0f}h{minutes:.0f}m"

def get_gpu_memory():
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3
        return memory_allocated, memory_reserved
    return 0, 0

def setup_distributed(rank, world_size, backend='nccl'):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    dist.destroy_process_group()

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.maft.model import MAFT
from model.maft.model.coupled_maft import CoupledMAFT, build_coupled_maft
from data import build_dataset, build_dataloader
from utils.logger import get_logger


class MAFTTrainerDDP:

    def __init__(self, config_path, exp_name=None, resume=None, rank=0, world_size=1):
        self.rank = rank
        self.world_size = world_size
        self.config_path = config_path

        self.config = self._load_config(config_path)

        self.exp_name = exp_name or self.config.get('output', {}).get('exp_name', 'maft_exp')
        self.exp_dir = os.path.join('experiments', self.exp_name)
        if is_main_process():
            os.makedirs(self.exp_dir, exist_ok=True)
            print(f"Experiment directory: {self.exp_dir}")

        self.device = torch.device(f'cuda:{rank}')

        if is_main_process():
            self.logger = get_logger(
                name='MAFT',
                log_file=os.path.join(self.exp_dir, 'train.log'),
                log_level=logging.INFO
            )

            self.logger.info(f"Using device: {self.device}, World size: {world_size}")

            self._init_wandb()
        else:
            self.logger = None

        self._set_random_seed(self.config['train']['seed'] + rank)

        self._build_model()

        self._build_dataloader()

        self._build_optimizer()

        self.current_epoch = 0
        self.best_metric = 0.0
        self.best_val_loss = float('inf')
        self.global_step = 0

        self.train_start_time = None
        self.epoch_start_time = None
        self.batch_times = []

        if resume:
            self._resume_training(resume)

    def _load_config(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config

    def _set_random_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _build_model(self):
        if is_main_process():
            self.logger.info("Building MAFT model...")

        model_name = self.config['model'].pop("name", "MAFT")
        maft_model = MAFT(**self.config['model'])
        maft_model = maft_model.to(self.device)

        coupling_cfg = self.config.get('coupling', {})
        coupling_enabled = coupling_cfg.get('enabled', False)

        if coupling_enabled:
            if is_main_process():
                self.logger.info("🔗 Building CoupledMAFT with bidirectional coupling...")

            self.model = build_coupled_maft(
                maft_model=maft_model,
                config=self.config,
                device=str(self.device)
            )
            self.coupling_enabled = True

            if is_main_process():
                self.logger.info(f"   Diffusion weight: {self.model.diff_weight}")
                self.logger.info(f"   Shape consistency weight: {self.model.sc_weight}")
                self.logger.info(f"   Dedup weight: {self.model.dedup_weight}")
        else:
            self.model = maft_model
            self.coupling_enabled = False

        self._load_pretrained_weights()

        if self.world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[self.rank],
                output_device=self.rank,
                find_unused_parameters=True
            )

        if is_main_process():
            raw_model = self.model.module if hasattr(self.model, 'module') else self.model
            total_params = sum(p.numel() for p in raw_model.parameters())
            trainable_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
            self.logger.info(f"Total parameters: {total_params:,}")
            self.logger.info(f"Trainable parameters: {trainable_params:,}")

            wandb.config.update({
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'model_size_mb': sum(p.numel() * p.element_size() for p in raw_model.parameters()) / 1024**2,
                'coupling_enabled': coupling_enabled
            })

            try:
                wandb.watch(raw_model, log="all", log_freq=1000, log_graph=True)
                self.logger.info("✅ Model architecture logged to wandb")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to log model to wandb: {e}")

    def _load_pretrained_weights(self):
        train_config = self.config['train']
        pretrain_path = train_config.get('pretrain')

        if pretrain_path and os.path.exists(pretrain_path):
            if is_main_process():
                self.logger.info(f"Loading MAFT pretrained weights from {pretrain_path}")

            checkpoint = torch.load(pretrain_path, map_location=self.device, weights_only=False)

            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            if self.coupling_enabled:
                target_model = self.model.maft
                if is_main_process():
                    self.logger.info("  Loading into CoupledMAFT.maft")
            else:
                target_model = self.model

            model_state = target_model.state_dict()
            filtered_state = {}
            skipped_keys = []
            for k, v in state_dict.items():
                if k in model_state:
                    if v.shape == model_state[k].shape:
                        filtered_state[k] = v
                    else:
                        skipped_keys.append(f"{k}: ckpt {v.shape} vs model {model_state[k].shape}")
                else:
                    skipped_keys.append(f"{k}: not in model")

            if is_main_process() and skipped_keys:
                self.logger.warning(f"  Skipped {len(skipped_keys)} keys with size mismatch")
                for sk in skipped_keys[:5]:
                    self.logger.warning(f"    {sk}")

            try:
                missing, unexpected = target_model.load_state_dict(filtered_state, strict=False)
                if is_main_process():
                    if missing:
                        self.logger.info(f"  Missing keys: {len(missing)} (coupling modules)")
                    self.logger.info(f"✅ Pretrained weights loaded successfully ({len(filtered_state)} keys)")
            except Exception as e:
                if is_main_process():
                    self.logger.error(f"❌ Failed to load pretrained weights: {e}")

    def _build_dataloader(self):
        if is_main_process():
            self.logger.info("Building data loaders...")

        train_dataset = build_dataset(self.config['data']['train'], self.logger if is_main_process() else None)

        dataloader_config = self.config['dataloader']['train'].copy()
        self.train_loader = build_dataloader(
            train_dataset,
            training=True,
            dist=(self.world_size > 1),
            **dataloader_config
        )

        val_dataset = build_dataset(self.config['data']['val'], self.logger if is_main_process() else None)

        val_dataloader_config = self.config['dataloader']['val'].copy()
        self.val_loader = build_dataloader(
            val_dataset,
            training=False,
            dist=(self.world_size > 1),
            **val_dataloader_config
        )

        if is_main_process():
            self.logger.info(f"Train samples: {len(train_dataset)}")
            self.logger.info(f"Val samples: {len(val_dataset)}")

            wandb.config.update({
                'train_samples': len(train_dataset),
                'val_samples': len(val_dataset),
                'train_batches': len(self.train_loader),
                'val_batches': len(self.val_loader)
            })

    def _build_optimizer(self):
        param_groups = self._get_param_groups()

        optimizer_config = self.config['optimizer']
        if optimizer_config['type'] == 'AdamW':
            self.optimizer = optim.AdamW(
                param_groups,
                lr=optimizer_config['lr'],
                weight_decay=optimizer_config['weight_decay']
            )
        elif optimizer_config['type'] == 'Adam':
            self.optimizer = optim.Adam(
                param_groups,
                lr=optimizer_config['lr'],
                weight_decay=optimizer_config['weight_decay']
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_config['type']}")

        scheduler_config = self.config['lr_scheduler']
        if scheduler_config['type'] == 'PolyLR':
            self.scheduler = optim.lr_scheduler.PolynomialLR(
                self.optimizer,
                total_iters=scheduler_config['max_iters'],
                power=scheduler_config['power']
            )
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_config['type']}")

    def _get_param_groups(self):
        optimizer_config = self.config['optimizer']

        return list(self.model.parameters())

    def train(self):
        total_epochs = self.config['train']['epochs']
        strategy_config = self.config.get('training_strategy', {})

        if is_main_process():
            self.logger.info("=" * 80)
            self.logger.info("🚀 MAFT DISTRIBUTED TRAINING STARTED")
            self.logger.info("=" * 80)
            self.logger.info(f"📅 Start Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info(f"🏷️  Experiment: {self.exp_name}")
            self.logger.info(f"📊 Total Epochs: {total_epochs}")
            self.logger.info(f"💾 Devices: {self.world_size} x GPU")

            self.logger.info(f"📚 Train Samples: {len(self.train_loader.dataset):,}")
            self.logger.info(f"📚 Val Samples: {len(self.val_loader.dataset):,}")
            self.logger.info(f"🔢 Batch Size per GPU: {self.config['dataloader']['train']['batch_size']}")
            self.logger.info(f"🔢 Total Batch Size: {self.config['dataloader']['train']['batch_size'] * self.world_size}")

            mem_alloc, mem_reserved = get_gpu_memory()
            self.logger.info(f"💾 GPU Memory: {mem_alloc:.1f}GB / {mem_reserved:.1f}GB")
            self.logger.info("=" * 80)

        self.train_start_time = time.time()

        for epoch in range(self.current_epoch, total_epochs):
            self.current_epoch = epoch

            if self.world_size > 1 and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)

            if self.coupling_enabled:
                raw_model = self.model.module if hasattr(self.model, 'module') else self.model
                raw_model.update_coupling_weights(epoch)
                if is_main_process() and epoch % 10 == 0:
                    self.logger.info(f"🔗 Coupling weights updated: diff_weight={raw_model.diff_weight:.4f}")

            train_metrics = self._train_epoch()

            val_metrics = None
            if (epoch + 1) % self.config['evaluation']['eval_interval'] == 0:
                val_metrics = self._validate_epoch()

                if is_main_process():
                    if self._is_best_model(val_metrics):
                        self.best_metric = val_metrics.get('all_ap', 0.0)
                        self._save_checkpoint(is_best=True, eval_metrics=val_metrics)

                    if self._is_best_val_loss(val_metrics):
                        self.best_val_loss = val_metrics.get('val_loss', float('inf'))
                        self._save_best_val_loss_checkpoint(val_metrics)

            if is_main_process() and (epoch + 1) % self.config['train']['interval'] == 0:
                self._save_checkpoint(is_best=False)

            self.scheduler.step()

            if is_main_process():
                self._log_epoch_metrics(train_metrics, val_metrics)

        if is_main_process():
            self._save_latest_checkpoint()
            self.logger.info("Training completed!")
            self._create_training_summary()
            wandb.finish()

    def _train_epoch(self):
        self.model.train()
        self.epoch_start_time = time.time()

        total_loss = 0.0
        loss_components = {}
        num_batches = len(self.train_loader)

        for batch_idx, batch in enumerate(self.train_loader):
            batch_start_time = time.time()

            batch = self._move_batch_to_device(batch)

            self.optimizer.zero_grad()

            loss, loss_dict = self.model(batch, mode='loss')

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

            if isinstance(loss_dict, dict):
                for key, value in loss_dict.items():
                    if key not in loss_components:
                        loss_components[key] = 0.0
                    if isinstance(value, torch.Tensor):
                        loss_components[key] += value.item()
                    else:
                        loss_components[key] += value

            batch_time = time.time() - batch_start_time
            self.batch_times.append(batch_time)

            if is_main_process() and batch_idx % 50 == 0:
                self._log_batch_progress(batch_idx, num_batches, loss.item(), loss_dict, batch_time)

        if self.world_size > 1:
            total_loss_tensor = torch.tensor(total_loss, device=self.device)
            dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = total_loss_tensor.item() / (num_batches * self.world_size)

            avg_loss_components = {}
            for key, value in loss_components.items():
                value_tensor = torch.tensor(value, device=self.device)
                dist.all_reduce(value_tensor, op=dist.ReduceOp.SUM)
                avg_loss_components[key] = value_tensor.item() / (num_batches * self.world_size)
        else:
            avg_loss = total_loss / num_batches
            avg_loss_components = {k: v/num_batches for k, v in loss_components.items()}

        return {'loss': avg_loss, 'loss_components': avg_loss_components}

    def _log_batch_progress(self, batch_idx, num_batches, loss, loss_dict, batch_time):
        progress = (batch_idx + 1) / num_batches * 100
        current_lr = self.optimizer.param_groups[0]['lr']
        mem_alloc, mem_reserved = get_gpu_memory()

        if len(self.batch_times) > 10:
            avg_batch_time = np.mean(self.batch_times[-50:])
            remaining_batches = num_batches - batch_idx - 1
            eta_epoch = remaining_batches * avg_batch_time

            total_epochs = self.config['train']['epochs']
            if self.current_epoch > 0:
                elapsed_epochs = self.current_epoch + progress / 100
                avg_epoch_time = (time.time() - self.train_start_time) / elapsed_epochs
                remaining_epochs = total_epochs - elapsed_epochs
                eta_total = remaining_epochs * avg_epoch_time
            else:
                eta_total = eta_epoch * total_epochs
        else:
            eta_epoch = 0
            eta_total = 0

        loss_str = f"Loss: {loss:.4f}"
        if isinstance(loss_dict, dict) and loss_dict:
            loss_components = []
            for key, value in loss_dict.items():
                if isinstance(value, torch.Tensor):
                    loss_components.append(f"{key}: {value.item():.3f}")
                else:
                    loss_components.append(f"{key}: {value:.3f}")
            if loss_components:
                loss_str += f" ({', '.join(loss_components)})"

        self.logger.info(
            f"[E{self.current_epoch:03d}][{batch_idx:04d}/{num_batches:04d}] "
            f"{progress:5.1f}% | {loss_str} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {batch_time:.2f}s | "
            f"GPU{self.rank}: {mem_alloc:.1f}GB"
        )

        if batch_idx % 100 == 0 and eta_total > 0:
            eta_epoch_str = format_time(eta_epoch)
            eta_total_str = format_time(eta_total)
            eta_finish = datetime.datetime.now() + datetime.timedelta(seconds=eta_total)

            self.logger.info(
                f"     ⏰ ETA: {eta_epoch_str} (epoch) | {eta_total_str} (total) | "
                f"Finish: {eta_finish.strftime('%m-%d %H:%M')}"
            )

        if batch_idx % 50 == 0:
            batch_metrics = {
                'batch/loss': loss,
                'batch/learning_rate': current_lr,
                'batch/batch_time': batch_time,
                'batch/gpu_memory_allocated': mem_alloc,
                'batch/gpu_memory_reserved': mem_reserved,
                'batch/epoch': self.current_epoch,
                'batch/progress': progress,
            }

            if isinstance(loss_dict, dict) and loss_dict:
                for key, value in loss_dict.items():
                    if isinstance(value, torch.Tensor):
                        batch_metrics[f'batch/loss_{key}'] = value.item()
                    else:
                        batch_metrics[f'batch/loss_{key}'] = value

            wandb.log(batch_metrics, step=self.global_step)

        self.global_step += 1

    def _validate_epoch(self):
        if is_main_process():
            self.logger.info('Validation')

        self.model.eval()

        pred_insts, gt_insts = [], []
        total_loss = 0.0
        num_batches = len(self.val_loader)

        accumulated_loss_components = {}

        if is_main_process():
            from tqdm import tqdm
            progress_bar = tqdm(total=len(self.val_loader), desc="Validating")

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                batch = self._move_batch_to_device(batch)

                loss, loss_dict = self.model(batch, mode='loss')
                total_loss += loss.item()

                if isinstance(loss_dict, dict):
                    for key, value in loss_dict.items():
                        if isinstance(value, torch.Tensor):
                            value = value.item()
                        if key not in accumulated_loss_components:
                            accumulated_loss_components[key] = 0.0
                        accumulated_loss_components[key] += value

                try:
                    result = self.model(batch, mode='predict')

                    pred_insts.append(result['pred_instances'])
                    gt_insts.append(result['gt_instances'])
                except Exception as e:
                    if is_main_process() and batch_idx == 0:
                        self.logger.warning(f"Model does not support 'predict' mode: {e}")
                    pred_insts, gt_insts = [], []
                    break

                if is_main_process():
                    progress_bar.update()

        if is_main_process():
            progress_bar.close()

        if self.world_size > 1:
            total_loss_tensor = torch.tensor(total_loss, device=self.device)
            dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = total_loss_tensor.item() / (num_batches * self.world_size)
        else:
            avg_loss = total_loss / num_batches

        avg_loss_components = {}
        for key, value in accumulated_loss_components.items():
            avg_loss_components[key] = value / num_batches

        eval_res = {
            'val_loss': avg_loss,
            'all_ap': 0.0,
            'all_ap_50%': 0.0,
            'all_ap_25%': 0.0,
            'loss_components': avg_loss_components if avg_loss_components else None
        }

        if is_main_process() and len(pred_insts) > 0:
            try:
                from model.maft.evaluation import ScanNetEval
                from utils.consts import RFS_labels
                val_dataset = self.val_loader.dataset
                classes = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf',
                           'picture', 'counter', 'desk', 'curtain', 'refridgerator', 'shower curtain', 'toilet', 'sink', 'bathtub',
                           'otherfurniture', 'kitchen_cabinet', 'display', 'trash_bin', 'other_shelf', 'other_table']

                self.logger.info('Evaluate instance segmentation')
                scannet_eval = ScanNetEval(classes)
                eval_res_ap = scannet_eval.evaluate(pred_insts, gt_insts)

                eval_res.update({
                    'all_ap': eval_res_ap['all_ap'],
                    'all_ap_50%': eval_res_ap['all_ap_50%'],
                    'all_ap_25%': eval_res_ap['all_ap_25%']
                })

                if 'classes' in eval_res_ap:
                    eval_res['classes_ap'] = eval_res_ap['classes']
                if 'classes_ap_50%' in eval_res_ap:
                    eval_res['classes_ap_50%'] = eval_res_ap['classes_ap_50%']
                if 'classes_ap_25%' in eval_res_ap:
                    eval_res['classes_ap_25%'] = eval_res_ap['classes_ap_25%']

                eval_res['class_names'] = classes

                self.logger.info('AP: {:.3f}. AP_50: {:.3f}. AP_25: {:.3f}'.format(
                    eval_res['all_ap'], eval_res['all_ap_50%'], eval_res['all_ap_25%']))

                if 'classes_ap' in eval_res:
                    self.logger.info("Per-class AP:")
                    for i, (class_name, ap_val) in enumerate(zip(classes, eval_res['classes_ap'])):
                        self.logger.info(f"  {class_name}: {ap_val:.3f}")

            except Exception as e:
                self.logger.warning(f"AP evaluation failed: {e}")

        if self.world_size > 1:
            for key in ['all_ap', 'all_ap_50%', 'all_ap_25%']:
                value_tensor = torch.tensor(eval_res[key], device=self.device)
                dist.broadcast(value_tensor, src=0)
                eval_res[key] = value_tensor.item()

        return eval_res

    def _move_batch_to_device(self, batch):
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device)
        return batch

    def _is_best_model(self, metrics):
        current_metric = metrics.get('all_ap', 0.0)
        return current_metric > self.best_metric

    def _is_best_val_loss(self, metrics):
        current_val_loss = metrics.get('val_loss', float('inf'))
        return current_val_loss < self.best_val_loss

    def _save_checkpoint(self, is_best=False, eval_metrics=None):
        if not is_main_process():
            return

        model_state_dict = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()

        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metric': self.best_metric,
            'global_step': self.global_step,
            'config': self.config
        }

        if is_best and eval_metrics:
            if hasattr(self, 'best_model_path') and os.path.exists(self.best_model_path):
                os.remove(self.best_model_path)

            ap = eval_metrics.get('all_ap', 0.0)
            ap_50 = eval_metrics.get('all_ap_50%', 0.0)
            ap_25 = eval_metrics.get('all_ap_25%', 0.0)

            best_filename = f'epoch{self.current_epoch:03d}_AP_{ap:.4f}_{ap_50:.4f}_{ap_25:.4f}.pth'
            self.best_model_path = os.path.join(self.exp_dir, best_filename)
            torch.save(checkpoint, self.best_model_path)

            self.logger.info(f"💾 Best model saved locally (wandb upload disabled)")

            self.logger.info(f"🏆 Saved best model: {best_filename}")
            self.logger.info(f"    AP: {ap:.4f}, AP_50: {ap_50:.4f}, AP_25: {ap_25:.4f}")

            wandb.log({
                'best_model/epoch': self.current_epoch,
                'best_model/ap': ap,
                'best_model/ap_50': ap_50,
                'best_model/ap_25': ap_25,
                'best_model/filename': best_filename
            }, step=self.global_step)
        elif is_best:
            best_path = os.path.join(self.exp_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model at epoch {self.current_epoch}")

    def _save_latest_checkpoint(self):
        if not is_main_process():
            return

        model_state_dict = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()

        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metric': self.best_metric,
            'global_step': self.global_step,
            'config': self.config
        }

        latest_path = os.path.join(self.exp_dir, 'latest.pth')
        torch.save(checkpoint, latest_path)
        self.logger.info(f"💾 Saved latest model: latest.pth (epoch {self.current_epoch})")

    def _save_best_val_loss_checkpoint(self, eval_metrics):
        if not is_main_process():
            return

        model_state_dict = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()

        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metric': self.best_metric,
            'best_val_loss': self.best_val_loss,
            'global_step': self.global_step,
            'config': self.config
        }

        best_model_path = os.path.join(self.exp_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            os.remove(best_model_path)

        torch.save(checkpoint, best_model_path)

        val_loss = eval_metrics.get('val_loss', float('inf'))
        self.logger.info(f"🏆 Saved best val loss model: best_model.pth (epoch {self.current_epoch}, val_loss: {val_loss:.4f})")

        wandb.log({
            'best_val_loss_model/epoch': self.current_epoch,
            'best_val_loss_model/val_loss': val_loss,
            'best_val_loss_model/ap': eval_metrics.get('all_ap', 0.0)
        }, step=self.global_step)

    def _resume_training(self, resume_path):
        if is_main_process():
            self.logger.info(f"Resuming training from {resume_path}")

        checkpoint = torch.load(resume_path, map_location=self.device, weights_only=False)

        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint['epoch'] + 1
        self.best_metric = checkpoint['best_metric']

        if 'best_val_loss' in checkpoint:
            self.best_val_loss = checkpoint['best_val_loss']
        else:
            self.best_val_loss = float('inf')

        if 'global_step' in checkpoint:
            self.global_step = checkpoint['global_step']
        else:
            self.global_step = self.current_epoch * len(self.train_loader)

        if is_main_process():
            wandb.config.update({
                'resumed_from_epoch': checkpoint['epoch'],
                'resumed_best_metric': checkpoint['best_metric'],
                'resumed_global_step': self.global_step,
                'resume_checkpoint': resume_path
            })
            self.logger.info(f"Resumed training: Epoch {self.current_epoch}, Best AP: {self.best_metric:.4f}, Global Step: {self.global_step}")

    def _log_epoch_metrics(self, train_metrics, val_metrics=None):
        epoch_time = time.time() - self.epoch_start_time if self.epoch_start_time else 0
        total_time = time.time() - self.train_start_time if self.train_start_time else 0

        total_epochs = self.config['train']['epochs']
        if self.current_epoch > 0:
            avg_epoch_time = total_time / (self.current_epoch + 1)
            remaining_epochs = total_epochs - self.current_epoch - 1
            eta_total = remaining_epochs * avg_epoch_time
        else:
            eta_total = 0

        mem_alloc, mem_reserved = get_gpu_memory()

        train_loss_str = f"Train Loss: {train_metrics['loss']:.4f}"
        if 'loss_components' in train_metrics and train_metrics['loss_components']:
            components = [f"{k}: {v:.3f}" for k, v in train_metrics['loss_components'].items()]
            train_loss_str += f" ({', '.join(components)})"

        progress = (self.current_epoch + 1) / total_epochs * 100

        self.logger.info("=" * 80)
        self.logger.info(
            f"🎯 EPOCH {self.current_epoch:03d}/{total_epochs:03d} COMPLETED "
            f"({progress:5.1f}%) | GPUs: {self.world_size}"
        )
        self.logger.info(f"⏱️  Time: {format_time(epoch_time)} (epoch) | {format_time(total_time)} (total)")

        if eta_total > 0:
            eta_finish = datetime.datetime.now() + datetime.timedelta(seconds=eta_total)
            self.logger.info(f"⏰ ETA: {format_time(eta_total)} | Finish: {eta_finish.strftime('%Y-%m-%d %H:%M:%S')}")

        self.logger.info(f"📊 {train_loss_str}")

        if val_metrics:
            val_loss_str = f"Val Loss: {val_metrics['val_loss']:.4f}"
            ap_str = f"AP: {val_metrics['all_ap']:.4f} | AP_50: {val_metrics['all_ap_50%']:.4f} | AP_25: {val_metrics['all_ap_25%']:.4f}"
            self.logger.info(f"📈 {val_loss_str} | {ap_str}")

            if val_metrics['all_ap'] > self.best_metric:
                self.logger.info("🏆 NEW BEST AP MODEL!")
            if val_metrics['val_loss'] < self.best_val_loss:
                self.logger.info("🏆 NEW BEST VAL LOSS MODEL!")

        current_lr = self.optimizer.param_groups[0]['lr']
        self.logger.info(f"🔧 LR: {current_lr:.2e} | GPU{self.rank}: {mem_alloc:.1f}GB/{mem_reserved:.1f}GB")

        epoch_metrics = {
            'epoch': self.current_epoch,
            'train/loss': train_metrics['loss'],
            'train/learning_rate': current_lr,

            'time/epoch_time': epoch_time,
            'time/total_time': total_time,
            'time/avg_epoch_time': total_time / (self.current_epoch + 1) if self.current_epoch > 0 else epoch_time,
            'time/eta_total': eta_total,
            'time/progress': progress,

            'system/gpu_memory_allocated': mem_alloc,
            'system/gpu_memory_reserved': mem_reserved,
            'system/best_metric': self.best_metric,
        }

        if 'loss_components' in train_metrics and train_metrics['loss_components']:
            for key, value in train_metrics['loss_components'].items():
                epoch_metrics[f'train/loss_{key}'] = value

        if val_metrics:
            epoch_metrics.update({
                'val/loss': val_metrics['val_loss'],
                'val/all_ap': val_metrics['all_ap'],
                'val/all_ap_50': val_metrics['all_ap_50%'],
                'val/all_ap_25': val_metrics['all_ap_25%'],
                'val/is_best': val_metrics['all_ap'] > self.best_metric,
            })

            if 'loss_components' in val_metrics and val_metrics['loss_components']:
                for key, value in val_metrics['loss_components'].items():
                    epoch_metrics[f'val/loss_{key}'] = value

            if 'classes_ap' in val_metrics and 'class_names' in val_metrics:
                class_names = val_metrics['class_names']
                for i, (class_name, ap_val) in enumerate(zip(class_names, val_metrics['classes_ap'])):
                    epoch_metrics[f'val/class_ap/{class_name}'] = ap_val

                if 'classes_ap_50%' in val_metrics:
                    for i, (class_name, ap_val) in enumerate(zip(class_names, val_metrics['classes_ap_50%'])):
                        epoch_metrics[f'val/class_ap_50/{class_name}'] = ap_val

                if 'classes_ap_25%' in val_metrics:
                    for i, (class_name, ap_val) in enumerate(zip(class_names, val_metrics['classes_ap_25%'])):
                        epoch_metrics[f'val/class_ap_25/{class_name}'] = ap_val

            for key, value in val_metrics.items():
                if key.startswith('classes_') and key not in ['classes_ap', 'classes_ap_50%', 'classes_ap_25%']:
                    epoch_metrics[f'val/{key}'] = value

        if self.batch_times:
            recent_batch_times = self.batch_times[-100:]
            epoch_metrics.update({
                'time/avg_batch_time': np.mean(recent_batch_times),
                'time/batch_time_std': np.std(recent_batch_times),
                'time/max_batch_time': np.max(recent_batch_times),
                'time/min_batch_time': np.min(recent_batch_times),
            })

        wandb.log(epoch_metrics, step=self.global_step)

        wandb.log({'lr_curve': current_lr}, step=self.global_step)

    def _init_wandb(self):
        model_config = self.config['model'].copy()
        train_config = self.config['train'].copy()

        wandb_config = {
            'model': model_config,
            'architecture': 'MAFT',

            'epochs': train_config.get('epochs', 100),
            'batch_size': self.config['dataloader']['train']['batch_size'],
            'effective_batch_size': self.config['dataloader']['train']['batch_size'] * self.world_size,
            'learning_rate': self.config['optimizer']['lr'],
            'weight_decay': self.config['optimizer']['weight_decay'],
            'optimizer': self.config['optimizer']['type'],
            'scheduler': self.config['lr_scheduler']['type'],
            'seed': train_config.get('seed', 42),

            'world_size': self.world_size,
            'device': str(self.device),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else 'N/A',
            'pytorch_version': torch.__version__,

            'dataset': self.config.get('data', {}),
            'dataloader': self.config.get('dataloader', {}),
        }

        wandb.init(
            project="MAFT-Training",
            name=self.exp_name,
            config=wandb_config,
            dir=self.exp_dir,
            resume="allow" if os.path.exists(os.path.join(self.exp_dir, 'wandb')) else None,
            tags=['distributed', 'instance_segmentation', 'maft'],
            notes=f""
        )

        try:
            if os.path.exists(self.config_path):
                wandb.save(self.config_path, base_path=os.path.dirname(self.config_path))
                self.logger.info(f"Config file uploaded to wandb: {self.config_path}")
            else:
                self.logger.warning(f"Config file not found, cannot upload: {self.config_path}")
        except Exception as e:
            self.logger.warning(f"Failed to upload config file: {e}")

        self.logger.info(f"🌟 Wandb initialized: {wandb.run.name}")
        self.logger.info(f"🔗 Wandb URL: {wandb.run.url}")

    def _create_training_summary(self):
        total_time = time.time() - self.train_start_time if self.train_start_time else 0

        summary_data = [
            ["Metric", "Value"],
            ["🏆 Best AP", f"{self.best_metric:.4f}"],
            ["📊 Total Epochs", f"{self.current_epoch + 1}"],
            ["⏱️ Total Time", format_time(total_time)],
            ["💾 World Size", f"{self.world_size}"],
            ["🔢 Batch Size", f"{self.config['dataloader']['train']['batch_size']}"],
            ["🔢 Effective Batch Size", f"{self.config['dataloader']['train']['batch_size'] * self.world_size}"],
            ["📚 Train Samples", f"{len(self.train_loader.dataset):,}"],
            ["📚 Val Samples", f"{len(self.val_loader.dataset):,}"],
        ]

        wandb.log({"training_summary": wandb.Table(data=summary_data, columns=["Metric", "Value"])})

        final_metrics = {
            "final/best_ap": self.best_metric,
            "final/total_epochs": self.current_epoch + 1,
            "final/total_time_hours": total_time / 3600,
            "final/avg_epoch_time_minutes": (total_time / (self.current_epoch + 1)) / 60,
        }

        wandb.log(final_metrics)

        self.logger.info("📋 Training Summary:")
        for row in summary_data[1:]:
            self.logger.info(f"   {row[0]}: {row[1]}")

        self.logger.info(f"Experiment completed! Best AP: {self.best_metric:.4f}")
        self.logger.info(f"Wandb URL: {wandb.run.url}")


def run_worker(rank, world_size, config_path, exp_name, resume):
    try:
        setup_distributed(rank, world_size)

        trainer = MAFTTrainerDDP(
            config_path=config_path,
            exp_name=exp_name,
            resume=resume,
            rank=rank,
            world_size=world_size
        )

        trainer.train()

    except Exception as e:
        print(f"Error in rank {rank}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description='MAFT Distributed Training')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--exp-name', type=str, help='Experiment name')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--world-size', type=int, default=2, help='Number of GPUs to use')

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    if torch.cuda.device_count() < args.world_size:
        raise RuntimeError(f"Need {args.world_size} GPUs, but only {torch.cuda.device_count()} available")

    if args.world_size == 1:
        trainer = MAFTTrainerDDP(
            config_path=args.config,
            exp_name=args.exp_name,
            resume=args.resume,
            rank=0,
            world_size=1
        )
        trainer.train()
    else:
        import torch.multiprocessing as mp
        mp.spawn(
            run_worker,
            args=(args.world_size, args.config, args.exp_name, args.resume),
            nprocs=args.world_size,
            join=True
        )


if __name__ == '__main__':
    main()
