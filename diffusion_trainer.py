# Python standard library
import os
import json
from typing import Dict, Tuple, Union, Optional

# Third party libraries
import numpy as np
import random
import torch
import torch.distributed as dist
import wandb
from omegaconf import DictConfig
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader
from tqdm import trange, tqdm
from transformers import AutoTokenizer
from omegaconf import OmegaConf
from optimi import StableAdamW
from torch_ema import ExponentialMovingAverage
from functools import partial

from encoder_trainer import EncoderTrainer
from architecture.score_estimator import ScoreEstimator
from estimation.metrics import compute_metric
from diffusion_utils.dynamic import DynamicSDE
from diffusion_utils.solvers import create_solver
from utils import DatasetDDP, reduce_tensor, BatchEncoding
from utils.diffusion_utils import get_stat, mse_loss
from utils.ddp_utils import gather_texts
from utils.logging_utils import config_to_wandb, log_batch_of_tensors_to_wandb, log_batch_of_texts_to_wandb
from estimation.fid import calculate_fid_for_embs
from estimation.util import truncate_text
from utils.pylogger import RankedLogger
from utils.sharded_dataset import ShardedDataset


class DiffusionTrainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.step = 0 

        self.logger = RankedLogger(name="trainer", rank_zero_only=False, rank=self.cfg.ddp.global_rank)

        # Initialize tokenizer and set vocab configs
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.autoencoder.model.text_encoder)
        self.vocab_size = self.tokenizer.vocab_size
        self.device = torch.device(f"cuda:{self.cfg.ddp.local_rank}") if self.cfg.ddp.enabled else torch.device("cuda")
        
        # Setup autoencoder
        self.autoencoder = EncoderTrainer(self.cfg)
        self.autoencoder.encoder.eval()
        self.autoencoder.decoder.eval()
        self.autoencoder.decoder.to("cpu")

        # Setup score estimator
        self.score_estimator = ScoreEstimator(self.cfg.diffusion)
        self.logger.info(f"Score estimator parameters: {sum(p.numel() for p in self.score_estimator.parameters() if p.requires_grad)}")
        self.score_estimator.to(self.device)

        # Setup dynamic
        self.dynamic = DynamicSDE(self.cfg)
        self.diff_eq_solver = create_solver(self.cfg)(
            dynamic=self.dynamic,
            score_fn=partial(self.calc_score, model=self.score_estimator),
            ode_sampling=self.cfg.diffusion.dynamic.ode_sampling
        )

        # Setup EMA
        self.ema = ExponentialMovingAverage(self.score_estimator.parameters(), decay=self.cfg.diffusion.ema.decay)

        if self.cfg.training == "diffusion":
            # Initialize training components
            self._setup_optimizer()
            self._setup_scheduler()
            
            is_loaded = self.load_checkpoint()

            # Setup DDP
            if dist.is_initialized() and self.cfg.ddp.enabled:
                self._setup_ddp()

            # Log parameter counts
            self._log_parameter_counts()
            
            self.device = torch.device(f"cuda:{self.cfg.ddp.local_rank}")
            if dist.is_initialized() and dist.get_rank() == 0:
                config_to_wandb(self.cfg)
            
            if is_loaded and dist.is_initialized() and self.cfg.ddp.enabled:
                self.estimate()
                self.validate()

    def _setup_ddp(self):
        """Setup Distributed Data Parallel."""
        self.ddp_score_estimator = self.score_estimator
        
        if self.cfg.ddp.enabled:
            self.ddp_score_estimator = torch.nn.parallel.DistributedDataParallel(
                self.score_estimator,
                device_ids=[self.cfg.ddp.local_rank],
                broadcast_buffers=False,
                find_unused_parameters=True,
            )

    def _setup_optimizer(self) -> None:
        self.grad_clip_norm = self.cfg.diffusion.optimizer.grad_clip_norm
        
        parameters = [par[1] for par in self.score_estimator.named_parameters() if par[1].requires_grad]
        
        if self.cfg.diffusion.optimizer.name == "adamw":
            optimizer = AdamW(
                parameters,
                lr=self.cfg.diffusion.optimizer.learning_rate,
                weight_decay=self.cfg.diffusion.optimizer.weight_decay,
                betas=(self.cfg.diffusion.optimizer.betas[0], self.cfg.diffusion.optimizer.betas[1]),
                eps=self.cfg.diffusion.optimizer.eps,
            )
        elif self.cfg.diffusion.optimizer.name == "stableadam":
            optimizer = StableAdamW(
                parameters,
                lr=self.cfg.diffusion.optimizer.learning_rate,
                weight_decay=self.cfg.diffusion.optimizer.weight_decay,
                betas=(self.cfg.diffusion.optimizer.betas[0], self.cfg.diffusion.optimizer.betas[1]),
                eps=self.cfg.diffusion.optimizer.eps,
            )
        
        self.optimizer = optimizer

    def _setup_scheduler(self) -> None:
        self.scheduler = CosineLRScheduler(
            self.optimizer,
            t_initial=self.cfg.diffusion.training.training_iters,
            lr_min=self.cfg.diffusion.optimizer.min_lr,
            warmup_lr_init=self.cfg.diffusion.optimizer.warmup_lr,
            warmup_t=self.cfg.diffusion.optimizer.linear_warmup,
            cycle_limit=1,
            t_in_epochs=False,
        )

    def _setup_train_data_generator(self) -> None:
        if hasattr(self, 'train_dataset'):
            del self.train_dataset

        if not hasattr(self, 'train_datasets_iter'):
            self.train_datasets_iter = DatasetDDP(
                config=self.cfg,
                split="train",
            ).get_dataset_iter()

        self.train_dataset = next(self.train_datasets_iter)
        self.logger.info(f"Dataset length: {len(self.train_dataset)}")

        self.train_loader = DataLoader(
            self.train_dataset,
            num_workers=self.cfg.diffusion.model.num_workers,
            batch_size=self.cfg.diffusion.training.batch_size_per_gpu,
            shuffle=True,
            collate_fn=self.collate_fn,
            drop_last=True,
        )

    def _setup_train_data_generator(self):
        if not hasattr(self, "train_dataset"):
            self.sharded_dataset = ShardedDataset(
                self.cfg, 
                "train", 
                self.cfg.dataset.prefetch_shards,
                self.logger
            )

        self.current_shard = self.sharded_dataset.get_next_shard()
        if self.current_shard is None:
            raise ValueError("No data available")
        
        self.logger.info(f"Loaded shard {self.sharded_dataset.current_shard_idx - 1}, "
                        f"size: {len(self.current_shard)}")
        self._create_dataloader_for_shard()
    
    def _create_dataloader_for_shard(self):
        if self.cfg.ddp.enabled:
            self.sampler_train = torch.utils.data.DistributedSampler(
                self.current_shard,
                shuffle=True,
            )
            self.sampler_train.set_epoch(self.step)
        else:
            self.sampler_train = None
        
        self.train_loader = DataLoader(
            dataset=self.current_shard,
            num_workers=self.cfg.diffusion.model.num_workers,
            batch_size=self.cfg.diffusion.training.batch_size_per_gpu,
            sampler=self.sampler_train,
            collate_fn=self.collate_fn,
            drop_last=True,
        )
        
        self.train_loader_iter = iter(self.train_loader)

    def _load_next_shard(self):
        """Loads the next data chunk"""
        # Clear memory from the current shard
        del self.current_shard
        del self.train_loader
        del self.train_loader_iter
        torch.cuda.empty_cache()
        
        # Load the next shard
        self.current_shard = self.sharded_dataset.get_next_shard()
        
        if self.current_shard is None:
            self.logger.info("All shards processed, restarting from beginning")
            self.sharded_dataset.reset()
            self.current_shard = self.sharded_dataset.get_next_shard()
        
        self.logger.info(f"Loaded shard {self.sharded_dataset.current_shard_idx - 1}, "
                        f"size: {len(self.current_shard)}")
        
        # Create a new DataLoader
        self._create_dataloader_for_shard()

    def _setup_valid_data_generator(self) -> None:
        if not hasattr(self, 'valid_dataset'):
            self.valid_dataset = ShardedDataset(
                self.cfg, 
                "test", 
                self.cfg.dataset.prefetch_shards,
                self.logger
            )

        self.current_valid_shard = self.valid_dataset.get_next_shard()
        if self.current_valid_shard is None:
            raise ValueError("No data available")
        
        self.logger.info(f"Loaded validation shard {self.valid_dataset.current_shard_idx - 1}, "
                        f"size: {len(self.current_valid_shard)}")

        if self.cfg.ddp.enabled:
            self.sampler_valid = torch.utils.data.DistributedSampler(
                self.current_valid_shard,
                shuffle=False,
            )
            self.sampler_valid.set_epoch(self.step)
        else:
            self.sampler_valid = None
        
        self.valid_loader = DataLoader(
            self.current_valid_shard,
            num_workers=self.cfg.diffusion.model.num_workers,
            batch_size=self.cfg.diffusion.training.batch_size_per_gpu,
            collate_fn=self.collate_fn,
            sampler=self.sampler_valid,
            drop_last=True,
        )

    def _log_parameter_counts(self) -> None:
        self.cfg.diffusion.params.score_estimator = sum(p.numel() for p in self.score_estimator.parameters() if p.requires_grad)

        all_params = dict()
        all_params["score_estimator"] = dict()
        for name, param in self.score_estimator.named_parameters():
            if param.requires_grad:
                all_params["score_estimator"][name] = param.numel()

        all_params = OmegaConf.create(all_params)
        self.cfg.diffusion.all_params = all_params

    def restore_checkpoint(self) -> None:
        path = os.path.join(self.cfg.project.checkpoint_dir, self.cfg.diffusion.model.load_checkpoint)
        state_dict = torch.load(path, map_location='cpu')
        self.score_estimator.load_state_dict(state_dict["score_estimator"])
        self.ema.load_state_dict(state_dict["ema"])
        self.autoencoder.latent_mean = state_dict["latent_mean"].to(self.device)
        self.autoencoder.latent_std = state_dict["latent_std"].to(self.device)
        self.logger.info(f"Checkpoint {self.cfg.diffusion.model.load_checkpoint} is restored")

    def load_checkpoint(self) -> None:
        if not self.cfg.diffusion.model.load_checkpoint:
            return False
        
        if isinstance(self.cfg.diffusion.model.load_checkpoint, str):
            path = os.path.join(self.cfg.project.checkpoint_dir, self.cfg.diffusion.model.load_checkpoint)
        else:
            path = self.find_last_checkpoint()
            if path is None:
                return False
        
        state_dict = torch.load(path, map_location='cpu')
        self.score_estimator.load_state_dict(state_dict["score_estimator"])
        self.ema.load_state_dict(state_dict["ema"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.scheduler.load_state_dict(state_dict["scheduler"])
        self.step = state_dict["step"]
        self.autoencoder.latent_mean = state_dict["latent_mean"].to(self.device)
        self.autoencoder.latent_std = state_dict["latent_std"].to(self.device)
        self.logger.info(f"Checkpoint {self.cfg.diffusion.model.load_checkpoint} loaded")
        return True
    
    def find_last_checkpoint(self) -> Optional[str]:
        prefix_folder = os.path.join(self.cfg.project.checkpoint_dir, self.cfg.diffusion.model.checkpoints_prefix)
        if not os.path.exists(prefix_folder):
            return None
        
        checkpoint_names = list(os.listdir(prefix_folder))
        checkpoint_names = [str(t).replace(".pth", "") for t in checkpoint_names]
        checkpoint_names = [int(t) for t in checkpoint_names if t.isdigit()]

        if not checkpoint_names:
            return None
            
        name = max(checkpoint_names)
        checkpoint_name = f"{prefix_folder}/{name}.pth"
        return checkpoint_name
        
    def save_checkpoint(self) -> None:
        if self.cfg.ddp.enabled and dist.get_rank() != 0:
            return

        if not os.path.exists(self.cfg.project.checkpoint_dir):
            os.makedirs(self.cfg.project.checkpoint_dir)
            
        prefix_folder = os.path.join(self.cfg.project.checkpoint_dir, self.cfg.diffusion.model.checkpoints_prefix)
        if not os.path.exists(prefix_folder):
            os.makedirs(prefix_folder)

        prefix = str(self.step)
        save_path = os.path.join(prefix_folder, prefix + ".pth")
        
        self.__save_checkpoint(save_path)
    
    def __save_checkpoint(self, save_path):
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "step": self.step,
                "score_estimator": self.score_estimator.state_dict(),
                "ema": self.ema.state_dict(),
                "latent_mean": self.autoencoder.latent_mean,
                "latent_std": self.autoencoder.latent_std,
            },
            save_path
        )
        self.logger.info(f"Save model to: {save_path}")

    def collate_fn(self, batch):
        texts = [sample["text_trg"] for sample in batch]

        tokenized_texts = self.tokenizer(
            texts,
            add_special_tokens=self.cfg.tokenizer.add_special_tokens,
            padding=self.cfg.tokenizer.padding,
            truncation=self.cfg.tokenizer.truncation,
            max_length=self.cfg.dataset.max_sequence_len,
            return_tensors=self.cfg.tokenizer.return_tensors,
            return_attention_mask=self.cfg.tokenizer.return_attention_mask,
            return_token_type_ids=self.cfg.tokenizer.return_token_type_ids,
        )

        new_batch = {
            "input_ids": tokenized_texts["input_ids"],
            "attention_mask": tokenized_texts["attention_mask"],
            "text_trg": texts,
        }

        return BatchEncoding(new_batch)

    def log_metric(self, metric_name: str, loader_name: str, value: Union[float, torch.Tensor, wandb.Image]):
        if not self.cfg.project.wandb_logging:
            return

        if wandb.run is None:
            return

        if dist.is_initialized() and dist.get_rank() == 0 or not self.cfg.ddp.enabled:
            wandb.log({f'{metric_name}/{loader_name}': value}, step=self.step)

    def optimizer_step(self, loss: torch.Tensor): 
        self.optimizer.zero_grad()
        
        parameters = [par[1] for par in self.score_estimator.named_parameters() if par[1].requires_grad]

        loss.backward()

        grad_norm = torch.norm(
            torch.stack([
                torch.norm(p.grad.detach(), 2) 
                for p in parameters 
                if p.grad is not None
            ]), 
            2
        )

        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                parameters,
                max_norm=self.grad_clip_norm
            )

        self.optimizer.step()

        # Update EMA
        self.ema.update(self.score_estimator.parameters())
        self.scheduler.step_update(self.step)
        
        score_estimator_weight_norm = torch.norm(
            torch.stack([
                torch.norm(p.data.detach(), 2) 
                for p in self.score_estimator.parameters()
            ]), 
            2
        )
        
        stat_dict = {
            "lr": self.optimizer.param_groups[0]['lr'],
            "grad_norm": grad_norm.item(),  
            "score_estimator-weight_norm": score_estimator_weight_norm.item(),
        }
        
        return stat_dict
    
    def log_data(self, total_loss, loss_dict = None, stat_dict = None, is_train: bool = True):
        if is_train:
            loader_name = "train_loader"
        else:
            loader_name = "valid_loader"
        
        # Total loss
        self.log_metric("Total_loss", loader_name, total_loss)

        # Losses
        if loss_dict is not None and is_train:
            for name in loss_dict:
                for k, v in loss_dict[name].items():
                    self.log_metric(loader_name, f"{name}-{k}", v)

        # Statistics
        if stat_dict is not None:
            for k, v in stat_dict.items():
                self.log_metric("statistics", k, v)
    
    def train(self) -> None:
        self.train_range = trange(self.step + 1, self.cfg.diffusion.training.training_iters + 1)
        self.train_loader_iter = iter([])
        self.logger.info("Training score estimator...")
        self._setup_train_data_generator()

        self.ddp_score_estimator.train()

        # Get latent statistics for specialized dataset
        self.autoencoder.latent_mean, self.autoencoder.latent_std = self.autoencoder.get_latent_statistics()

        for step in self.train_range:
            self.step = step
            
            batch = next(self.train_loader_iter, None)
            if batch is None:
                self._load_next_shard()
                batch = next(self.train_loader_iter, None)

            total_loss, loss_dict = self.calc_loss(batch)
            stat_dict = self.optimizer_step(total_loss)

            if self.step % self.cfg.diffusion.logging.log_freq == 0:
                if dist.is_initialized() and dist.get_rank() == 0:
                    self.log_data(total_loss, loss_dict, stat_dict, is_train=True)   
            
            self.train_range.set_description(f"total_loss: {total_loss.item():0.3f}")
            
            if self.step % self.cfg.diffusion.logging.save_freq == 0:
                self.save_checkpoint()
                
            if self.step % self.cfg.diffusion.logging.eval_freq == 0:
                self.validate()
                self.estimate()
                torch.cuda.empty_cache()

        self.save_checkpoint()

        if dist.is_initialized() and dist.get_rank() == 0:
            wandb.finish()

    @torch.no_grad()
    def validate(self) -> None:
        self._setup_valid_data_generator()
        self.ddp_score_estimator.eval()

        total_loss = torch.Tensor([0.0])
        valid_count = torch.Tensor([0.0])
        
        with self.ema.average_parameters():
            for batch in self.valid_loader:
                key = list(batch.keys())[0]
                batch_size = batch[key].shape[0]
                batch_loss, loss_dict = self.calc_loss(batch)
                
                valid_count += batch_size
                total_loss += batch_loss.item() * batch_size

        if self.cfg.ddp.enabled:
            valid_count = reduce_tensor(valid_count.cuda())
            total_loss = reduce_tensor(total_loss.cuda())
        total_loss = total_loss / valid_count
        
        self.log_data(total_loss, is_train=False)

        self.ddp_score_estimator.train()

    def sample_time(self, batch_size: int):
        T = self.cfg.diffusion.diffusion.T
        eps = self.cfg.diffusion.diffusion.eps
        return torch.cuda.FloatTensor(batch_size).uniform_() * (T - eps) + eps

    def calc_loss(self, batch) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        # Get latent
        batch = batch.to(self.device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16), torch.no_grad():
            encoder_latents, _ = self.autoencoder.get_latent(batch, bert_output_masking=False)
            clean_x = self.autoencoder.normalize_latent(encoder_latents)

        # Add noise to the clean latent
        batch_size = clean_x.size(0)

        t = self.sample_time(batch_size)
        marg_forward = self.dynamic.marginal(clean_x, t)
        x_t, noise = marg_forward['x_t'], marg_forward['noise']

        # self-cond estimate
        x_0_self_cond = torch.zeros_like(clean_x, dtype=clean_x.dtype)
        if self.cfg.diffusion.diffusion.use_self_cond and random.random() > 0.5:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16), torch.no_grad():
                x_0_self_cond = self.ddp_score_estimator(
                    x_t=x_t.clone(), 
                    time_t=t.clone(),
                    x_0_self_cond=x_0_self_cond
                ).detach()

        # model prediction
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            x_0 = self.ddp_score_estimator(
                x_t=x_t, 
                time_t=t,
                x_0_self_cond=x_0_self_cond
            )

        # MSE losses
        loss_x_0 = torch.mean(torch.square(clean_x - x_0))

        # Statistics
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16), torch.no_grad():
            loss_dict = {
                "clean_x": get_stat(clean_x.detach()),
                "x_0": get_stat(x_0.detach()),
                "x_t": get_stat(x_t.detach()),
            }

        return loss_x_0, loss_dict
    
    def calc_score(self, model, x_t, t, x_0_self_cond) -> Dict[str, torch.Tensor]:
        params = self.dynamic.marginal_params(t)
        x_0 = model(x_t=x_t, time_t=t, x_0_self_cond=x_0_self_cond)
        eps_theta = (x_t - params["mu"] * x_0) / params["std"]
        score = -eps_theta / params["std"]
        return {
            "score": score,
            "x_0": x_0,
            "eps_theta": eps_theta
        }

    @torch.no_grad()
    def generate_text(self, num_texts: int):
        self._setup_valid_data_generator()
        self.autoencoder.decoder.to(self.device)
        
        result_dict = {
            "GEN": [],
            "TRG": [],
        }
    
        for batch in self.valid_loader:
            batch = batch.to(self.device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                encoder_latents, _ = self.autoencoder.get_latent(batch, bert_output_masking=False)
                clean_x = self.autoencoder.normalize_latent(encoder_latents)

            gen_text, pred_embeddings, _ = self.generate_text_batch(batch_size=len(batch["input_ids"]))

            result_dict["TRG"] += batch["text_trg"]
            result_dict["GEN"] += gen_text

            if len(result_dict["TRG"]) >= num_texts:
                break

        self.autoencoder.decoder.to("cpu")
        return result_dict

    @torch.no_grad()
    def generate_text_batch(self, batch_size):
        pred_embeddings = self.pred_embeddings(batch_size=batch_size)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            pred_latents = self.autoencoder.denormalize_latent(pred_embeddings)
            pred_logits = self.autoencoder.decoder(pred_latents)

        text, _ = self.sample_from_logits(pred_logits)
        return text, pred_embeddings, pred_logits

    def sample_from_logits(self, logits):
        eos_token_id = self.tokenizer.eos_token_id 
        if eos_token_id is None:
            eos_token_id = self.tokenizer.sep_token_id
        
        tokens = logits.argmax(dim=-1).detach().cpu().tolist()
                    
        tokens_list = []
        for seq_list in tokens:
            for ind, token in enumerate(seq_list):
                if token == eos_token_id:
                    tokens_list.append(seq_list[:ind])
                    break
            else:
                tokens_list.append(seq_list)

        return self.tokenizer.batch_decode(tokens_list, skip_special_tokens=True), tokens

    @torch.no_grad()
    def pred_embeddings(self, batch_size) -> torch.Tensor:
        self.score_estimator.eval()

        num_latents = self.cfg.encoder.latent.num_latents
        shape = (
            batch_size,
            num_latents,
            self.cfg.encoder.latent.dim
        )

        x_t = self.dynamic.prior_sampling(shape).to(self.device)
        x_0_self_cond = torch.zeros_like(x_t)

        with self.ema.average_parameters(), torch.no_grad():
            timesteps = torch.linspace(
                self.cfg.diffusion.diffusion.T, 
                self.cfg.diffusion.diffusion.t_min, 
                self.cfg.diffusion.dynamic.N + 1, 
                device=self.device
            )
            
            for idx in tqdm(range(self.cfg.diffusion.dynamic.N)):
                t = timesteps[idx]
                next_t = timesteps[idx + 1]

                input_t = t * torch.ones(shape[0], device=self.device)
                next_input_t = next_t * torch.ones(shape[0], device=self.device)

                output = self.diff_eq_solver.step(
                    x_t=x_t, t=input_t, next_t=next_input_t,
                    x_0_self_cond=x_0_self_cond,
                )

                x_t, x_mean = output["x"], output["x_mean"]
                x_0_self_cond = output["x_0"]
        
        self.score_estimator.train()

        return x_mean

    @torch.no_grad()
    def estimate(self,):
        torch.cuda.empty_cache()

        # Generation
        self.logger.info("Generating texts...")
        if not self.cfg.ddp.enabled:
            num_texts = self.cfg.diffusion.generation.num_gen_texts
        else:
            num_texts = self.cfg.diffusion.generation.num_gen_texts // dist.get_world_size()
            if dist.get_rank() < self.cfg.diffusion.generation.num_gen_texts % dist.get_world_size():
                num_texts += 1
        
        result_dict = self.generate_text(num_texts=num_texts)

        # Gathering
        if self.cfg.ddp.enabled:
            for key in result_dict:
                result_dict[key] = gather_texts(result_dict[key])
                result_dict[key] = result_dict[key][:self.cfg.diffusion.generation.num_gen_texts]

        # Logging
        list_of_dicts = [{key: result_dict[key][i] for key in result_dict} for i in range(len(result_dict["TRG"]))]
                
        if not self.cfg.ddp.enabled or dist.get_rank() == 0:
            dir_path = os.path.join(self.cfg.project.path, self.cfg.diffusion.generation.texts_dir_path)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            prefix_folder = os.path.join(dir_path, self.cfg.diffusion.model.checkpoints_prefix)
            if not os.path.exists(prefix_folder):
                os.makedirs(prefix_folder)
            
            total_len = len(list_of_dicts)
            file_name = f"{self.step}-N={self.cfg.diffusion.dynamic.N}-len={total_len}.json"
            save_path = os.path.join(prefix_folder, file_name)
            json.dump(list_of_dicts, open(save_path, "w"), indent=4)
            self.logger.info(f"Texts are saved in {save_path}")

        # Metrics
        mauve_value = self._compute_mauve(result_dict["GEN"], result_dict["TRG"])
        ppl_value = self._compute_ppl(result_dict["GEN"])
        div_value = self._compute_div(result_dict["GEN"])

        if not self.cfg.ddp.enabled or dist.get_rank() == 0:
            dir_path = os.path.join(self.cfg.project.path, self.cfg.diffusion.generation.metrics_dir_path)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            prefix_folder = os.path.join(dir_path, self.cfg.diffusion.model.checkpoints_prefix)
            if not os.path.exists(prefix_folder):
                os.makedirs(prefix_folder)

            file_name = f"{self.step}-N={self.cfg.diffusion.dynamic.N}-len={total_len}.txt"
            save_path = os.path.join(prefix_folder, file_name)
            with open(save_path, "w") as f:
                f.write(f"Mauve: {mauve_value:0.5f}\n")
                f.write(f"PPL: {ppl_value:0.5f}\n")
                f.write(f"DIV: {div_value:0.5f}\n")

            self.logger.info(f"Metrics are saved in {save_path}")


        # self._compute_fid(result_dict["pred_latents"], result_dict["trg_latents"])

        torch.cuda.synchronize()

    def _compute_mauve(self, predictions, references):
        predictions = truncate_text(predictions, self.cfg.dataset.max_sequence_len, 1)
        references = truncate_text(references, self.cfg.dataset.max_sequence_len, 1)

        mauve_values = []
        size = self.cfg.diffusion.generation.num_texts_from_metric
        total_number = len(predictions) // size
        for ind_gen in range(total_number):
            for ind_trg in range(total_number):
                ind = ind_gen * total_number + ind_trg
                if ind % dist.get_world_size() == dist.get_rank():
                    mauve_values.append(
                        compute_metric(
                            "mauve", 
                            predictions=predictions[ind_gen * size:(ind_gen + 1) * size], 
                            references=references[ind_trg * size:(ind_trg + 1) * size]
                        )
                    )
        
        mauve_values = gather_texts(mauve_values)

        if dist.get_rank() == 0:
            mauve_values = np.array(mauve_values)
            mauve_value = np.mean(mauve_values)
            self.logger.info(f"Mauve: {mauve_value:0.5f}")
            self.log_metric(metric_name=f"{self.cfg.dataset.name}", loader_name="Mauve", value=mauve_value)
        
            return mauve_value
        else:
            return None

    def _compute_ppl(self, predictions):
        predictions = truncate_text(predictions, self.cfg.dataset.max_sequence_len, 1)

        ppl_values = []
        size = self.cfg.diffusion.generation.num_texts_from_metric
        total_number = len(predictions) // size
        for ind_gen in range(total_number):
            if ind_gen % dist.get_world_size() == dist.get_rank():
                ppl_values.append(
                    compute_metric(
                        "ppl", 
                        predictions=predictions[ind_gen * size:(ind_gen + 1) * size], 
                        references=None,
                    )
                )
        
        ppl_values = gather_texts(ppl_values)

        if dist.get_rank() == 0:
            ppl_values = np.array(ppl_values)
            ppl_value = np.mean(ppl_values)
            self.logger.info(f"PPL: {ppl_value:0.5f}")
            self.log_metric(metric_name=f"{self.cfg.dataset.name}", loader_name="PPL", value=ppl_value)
           
            return ppl_value
        else:
            return None

    def _compute_div(self, predictions):
        predictions = truncate_text(predictions, self.cfg.dataset.max_sequence_len, 1)

        div_values = []
        size = self.cfg.diffusion.generation.num_texts_from_metric
        total_number = len(predictions) // size
        for ind_gen in range(total_number):
            if ind_gen % dist.get_world_size() == dist.get_rank():
                div_values.append(
                    compute_metric(
                        "div", 
                        predictions=predictions[ind_gen * size:(ind_gen + 1) * size], 
                        references=None,
                    )
                )
        
        div_values = gather_texts(div_values)

        if dist.get_rank() == 0:
            div_values = np.array(div_values)
            div_value = np.mean(div_values)
            self.logger.info(f"DIV: {div_value:0.5f}")
            self.log_metric(metric_name=f"{self.cfg.dataset.name}", loader_name="DIV", value=div_value)

            return div_value
        else:
            return None

    def _compute_fid(self, pred_embeddings, trg_embeddings):
        if dist.get_rank() == 0:
            pred_embeddings = np.mean(np.array(pred_embeddings), axis=1)
            trg_embeddings = np.mean(np.array(trg_embeddings), axis=1)

            fid_value = calculate_fid_for_embs(pred_embeddings, trg_embeddings)
            self.logger.info(f"FID: {fid_value:0.5f}")
            self.log_metric(metric_name=f"{self.cfg.dataset.name}", loader_name="FID", value=fid_value)

            return fid_value
        else:
            return None

    def _compute_mem(self, predictions):
        predictions = truncate_text(predictions, self.cfg.dataset.max_sequence_len, 1)
        
        mem_values = []
        size = self.cfg.diffusion.generation.num_texts_from_metric
        total_number = len(predictions) // size
        for ind_gen in range(total_number):
            if ind_gen % dist.get_world_size() == dist.get_rank():
                mem_values.append(
                    compute_metric(
                        "mem", 
                        predictions=predictions[ind_gen * size:(ind_gen + 1) * size], 
                        train_unique_four_grams=self.cfg.diffusion.generation.train_unique_four_grams,
                        train_dataset_path=f"{self.cfg.dataset.dataset_path}/train",
                    )
                )
        
        mem_values = gather_texts(mem_values)

        if dist.get_rank() == 0:
            mem_values = np.array(mem_values)
            mem_value = np.mean(mem_values)
            self.logger.info(f"MEM: {mem_value:0.5f}")
            self.log_metric(metric_name=f"{self.cfg.dataset.name}", loader_name="MEM", value=mem_value)

            return mem_value
        else:
            return None
