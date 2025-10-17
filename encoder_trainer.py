# Python standard library
import os
import json
from typing import Dict, Tuple, Union

# Third party libraries
import numpy as np
import torch
import torch.distributed as dist
import wandb
from omegaconf import DictConfig
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.cuda.amp import GradScaler
from torch.nn.functional import cross_entropy
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader
from tqdm import trange, tqdm
from transformers import AutoTokenizer
from omegaconf import OmegaConf
from optimi import StableAdamW, Lion

from architecture.encoder import Encoder
from architecture.decoder import Decoder

from utils import DatasetDDP, BatchEncoding, reduce_tensor
from utils.sharded_dataset import ShardedDataset
from utils.logging_utils import config_to_wandb, log_batch_of_tensors_to_wandb 
from utils.pylogger import RankedLogger

from diffusion_utils.corruption import apply_corruption, prepare_corruption


def cross_entropy_loss(input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    loss = cross_entropy(input=input.reshape(-1, input.shape[-1]), target=target.reshape(-1), reduction="none")
    return (loss * mask.reshape(-1)).sum() / max(mask.sum(), 1)


def accuracy(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    pred = torch.argmax(logits, dim=-1)
    acc_tensor = (pred == target) * 1.
    acc = (acc_tensor * mask).sum() / max(mask.sum(), 1)
    return acc


def mse_loss_function(input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    loss = torch.mean((input - target) ** 2, dim=-1)
    return (loss * mask).sum() / max(mask.sum(), 1)
    

def to_str(list_of_tokens):
    return ",".join(str(t) for t in list_of_tokens)


def total_variation_loss(img):
     bs_img, h_img, w_img = img.size()
     tv_h = torch.pow(img[:,1:,:]-img[:,:-1,:], 2).sum()
     tv_w = torch.pow(img[:,:,1:]-img[:,:,:-1], 2).sum()
     return (tv_h+tv_w)/(bs_img*h_img*w_img)


def kl_divergence(latent):
    """
    latent: (batch_size, latent_dim)
    It's supposed to be a normal distribution with constant variance, so only mean is used.
    """
    return 0.5 * torch.mean(latent ** 2)


class EncoderTrainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.step = 0

        self.logger = RankedLogger(name="encoder_trainer", rank_zero_only=False, rank=self.cfg.ddp.global_rank)

        # Initialize tokenizer and set vocab configs
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.autoencoder.model.text_encoder)
        self.vocab_size = self.tokenizer.vocab_size

        self.device = torch.device(f"cuda:{self.cfg.ddp.local_rank}") if self.cfg.ddp.enabled else torch.device("cuda")
        
        # Configure encoder
        self._setup_encoder_cfg()
        self.encoder = Encoder(self.cfg.encoder).cuda()
        
        # Configure decoder cfg
        self._setup_decoder_cfg()
        self.decoder = Decoder(self.cfg.decoder).cuda()

        if self.cfg.training == "autoencoder":
            # Initialize training components
            self._setup_optimizer()
            self._setup_scheduler()
            self._setup_grad_scaler()

            is_loaded = self.load_checkpoint()

            # Log parameter counts
            self._log_parameter_counts()
        else:
            self.restore_checkpoint()
        
        # Setup DDP
        self._setup_ddp()

        if self.cfg.training == "autoencoder":
            if dist.is_initialized() and dist.get_rank() == 0:
                config_to_wandb(self.cfg)
            
            if is_loaded and dist.is_initialized() and self.cfg.ddp.enabled:
                self.validate()
        
    def _setup_encoder_cfg(self):
        """Setup encoder cfguration."""

        self.cfg.autoencoder.model.text_encoder = self.cfg.autoencoder.model.text_encoder
        self.cfg.encoder.model.text_encoder = self.cfg.autoencoder.model.text_encoder
        self.cfg.encoder.model.text_encoder_freeze_params = self.cfg.autoencoder.model.text_encoder_freeze_params
        self.cfg.encoder.tokens.vocab_size = self.vocab_size

    def _setup_decoder_cfg(self):
        """Setup decoder cfguration."""

        self.cfg.decoder.model.text_encoder = self.cfg.autoencoder.model.text_encoder
        self.cfg.decoder.model.text_encoder_freeze_params = self.cfg.autoencoder.model.text_encoder_freeze_params
        self.cfg.decoder.tokens.vocab_size = self.vocab_size
        self.cfg.decoder.tokens.mask_token_id = self.tokenizer.mask_token_id

    def _setup_ddp(self):
        """Setup Distributed Data Parallel."""
        self.ddp_encoder = self.encoder
        self.ddp_decoder = self.decoder
        
        if self.cfg.ddp.enabled:
            self.ddp_encoder = torch.nn.parallel.DistributedDataParallel(
                self.encoder,
                device_ids=[self.cfg.ddp.local_rank],
                broadcast_buffers=False,
                find_unused_parameters=True,
            )
            
            self.ddp_decoder = torch.nn.parallel.DistributedDataParallel(
                self.decoder,
                device_ids=[self.cfg.ddp.local_rank],
                broadcast_buffers=False,
                find_unused_parameters=True,
            )
        else:
            self.ddp_encoder = self.encoder
            self.ddp_decoder = self.decoder

    def _setup_optimizer(self) -> None:
        self.grad_clip_norm = self.cfg.autoencoder.optimizer.grad_clip_norm
        
        parameters_encoder = [par[1] for par in self.encoder.named_parameters() if par[1].requires_grad]
        parameters_decoder = [par[1] for par in self.decoder.named_parameters() if par[1].requires_grad]
        
        parameters = parameters_encoder + parameters_decoder
        
        if self.cfg.autoencoder.optimizer.name == "adamw":
            optimizer = AdamW(
                parameters,
                lr=self.cfg.autoencoder.optimizer.learning_rate,
                weight_decay=self.cfg.autoencoder.optimizer.weight_decay,
                betas=(self.cfg.autoencoder.optimizer.betas[0], self.cfg.autoencoder.optimizer.betas[1]),
                eps=self.cfg.autoencoder.optimizer.eps,
            )
        elif self.cfg.autoencoder.optimizer.name == "stableadam":
            optimizer = StableAdamW(
                parameters,
                lr=self.cfg.autoencoder.optimizer.learning_rate,
                weight_decay=self.cfg.autoencoder.optimizer.weight_decay,
                betas=(self.cfg.autoencoder.optimizer.betas[0], self.cfg.autoencoder.optimizer.betas[1]),
                eps=self.cfg.autoencoder.optimizer.eps,
            )
        elif self.cfg.autoencoder.optimizer.name == "lion":
            optimizer = Lion(
                parameters,
                lr=self.cfg.autoencoder.optimizer.learning_rate,
                weight_decay=self.cfg.autoencoder.optimizer.weight_decay,
                betas=(self.cfg.autoencoder.optimizer.betas[0], self.cfg.autoencoder.optimizer.betas[1]),
            )
        
        self.optimizer = optimizer

    def _setup_scheduler(self) -> None:
        self.scheduler = CosineLRScheduler(
            self.optimizer,
            t_initial=self.cfg.autoencoder.training.training_iters,
            lr_min=self.cfg.autoencoder.optimizer.min_lr,
            warmup_lr_init=self.cfg.autoencoder.optimizer.warmup_lr,
            warmup_t=self.cfg.autoencoder.optimizer.linear_warmup,
            cycle_limit=1,
            t_in_epochs=False,
        )
        
    def _setup_grad_scaler(self) -> None:
        self.grad_scaler = GradScaler()

    def _setup_train_data_generator(self) -> None:
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
            num_workers=self.cfg.autoencoder.model.num_workers,
            batch_size=self.cfg.autoencoder.training.batch_size_per_gpu,
            collate_fn=self.collate_fn,
            sampler=self.sampler_valid,
            drop_last=True,
        )
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
            num_workers=self.cfg.autoencoder.model.num_workers,
            batch_size=self.cfg.autoencoder.training.batch_size_per_gpu,
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

    def _log_parameter_counts(self) -> None:
        self.cfg.autoencoder.params.text_encoder = sum(p.numel() for p in self.encoder.text_encoder.parameters() if p.requires_grad)
        self.cfg.autoencoder.params.encoder = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad) - self.cfg.autoencoder.params.text_encoder
        self.cfg.autoencoder.params.decoder = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        self.cfg.autoencoder.params.total = self.cfg.autoencoder.params.text_encoder + self.cfg.autoencoder.params.encoder + self.cfg.autoencoder.params.decoder

    def load_checkpoint(self) -> None:
        if not self.cfg.autoencoder.model.load_checkpoint:
            return False
        
        if isinstance(self.cfg.autoencoder.model.load_checkpoint, str):
            path = os.path.join(self.cfg.project.checkpoint_dir, self.cfg.autoencoder.model.load_checkpoint)
        else:
            path = self.find_last_checkpoint()
            if path is None:
                return False
        
        state_dict = torch.load(path, map_location='cpu')
        self.encoder.load_state_dict(state_dict["encoder"])
        self.decoder.load_state_dict(state_dict["decoder"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.scheduler.load_state_dict(state_dict["scheduler"])
        self.grad_scaler.load_state_dict(state_dict["scaler"])
        self.step = state_dict["step"]
        self.latent_mean = state_dict["latent_mean"].to(self.device)
        self.latent_std = state_dict["latent_std"].to(self.device)
        self.encodings_mean = state_dict["encodings_mean"].to(self.device)
        self.encodings_std = state_dict["encodings_std"].to(self.device)
        self.logger.info(f"Checkpoint {self.cfg.autoencoder.model.load_checkpoint} loaded")
        return True
    
    def find_last_checkpoint(self) -> None:
        prefix_folder = os.path.join(self.cfg.project.checkpoint_dir, self.cfg.autoencoder.model.checkpoints_prefix)
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
    
    def restore_checkpoint(self) -> None:
        if not self.cfg.autoencoder.model.load_checkpoint:
            return
        
        path = os.path.join(self.cfg.project.checkpoint_dir, self.cfg.autoencoder.model.load_checkpoint)
        state_dict = torch.load(path, map_location='cpu')
        self.encoder.load_state_dict(state_dict["encoder"])
        self.decoder.load_state_dict(state_dict["decoder"])
        self.latent_mean = state_dict["latent_mean"].to(self.device)
        self.latent_std = state_dict["latent_std"].to(self.device)
        self.encodings_mean = state_dict["encodings_mean"].to(self.device)
        self.encodings_std = state_dict["encodings_std"].to(self.device)
        self.logger.info(f"Checkpoint {self.cfg.autoencoder.model.load_checkpoint} loaded")
        
    def save_checkpoint(self) -> None:
        if self.cfg.ddp.enabled and not dist.get_rank() == 0:
            return

        if not os.path.exists(self.cfg.project.checkpoint_dir):
            os.makedirs(self.cfg.project.checkpoint_dir)
            
        prefix_folder = os.path.join(self.cfg.project.checkpoint_dir, self.cfg.autoencoder.model.checkpoints_prefix)
        if not os.path.exists(prefix_folder):
            os.makedirs(prefix_folder)

        prefix = str(self.step)
        save_path = os.path.join(prefix_folder, prefix + ".pth")
        
        self.__save_checkpoint(save_path)
    
    def __save_checkpoint(self, save_path):
        state_dict = {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.grad_scaler.state_dict(),
            "step": self.step,
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
        }

        if self.latent_mean is not None and self.latent_std is not None:
            state_dict["latent_mean"] = self.latent_mean
            state_dict["latent_std"] = self.latent_std

        if hasattr(self, "encodings_mean") and hasattr(self, "encodings_std"):
            state_dict["encodings_mean"] = self.encodings_mean
            state_dict["encodings_std"] = self.encodings_std
            
        torch.save(
            state_dict,
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

        new_batch = {}
        new_batch["input_ids"] = tokenized_texts["input_ids"]
        new_batch["attention_mask"] = tokenized_texts["attention_mask"]

        # Make encodings masking and noising preparation
        new_batch["corrupted_attention_mask"], new_batch["mask"], new_batch["alpha"], new_batch["noise"] = prepare_corruption(
            encodings_shape=(new_batch["input_ids"].shape[0], new_batch["input_ids"].shape[1], self.cfg.encoder.embedding.dim),
            attention_mask=new_batch["attention_mask"],
            config=self.cfg.encoder.augmentation
        )

        return BatchEncoding(new_batch)

    def log_metric(self, metric_name: str, loader_name: str, value: Union[float, torch.Tensor, wandb.Image]):
        if dist.is_initialized() and dist.get_rank() == 0 or not self.cfg.ddp.enabled:
            wandb.log({f'{metric_name}/{loader_name}': value}, step=self.step)

    def optimizer_step(self, loss: torch.Tensor): 
        self.optimizer.zero_grad()
        
        parameters_encoder = [par[1] for par in self.encoder.named_parameters() if par[1].requires_grad]
        parameters_decoder = [par[1] for par in self.decoder.named_parameters() if par[1].requires_grad]
        parameters = parameters_encoder + parameters_decoder

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
        self.scheduler.step_update(self.step)
        
        stat_dict = {
            "lr": self.optimizer.param_groups[0]['lr'],
            "grad_norm": grad_norm.item(),  
        }
        
        return stat_dict
    
    def log_data(self, total_loss, loss_dict, stat_dict = None, is_train: bool = True):
        if not self.cfg.project.wandb_logging:
            return
        
        if is_train:
            loader_name = "train_loader"
        else:
            loader_name = "valid_loader"
        
        # Total loss
        self.log_metric("Total_loss", loader_name, total_loss)
        
        # Losses and accuracies
        for key in loss_dict:
            self.log_metric(key, loader_name, loss_dict[key])

        # Statistics
        if stat_dict is not None:
            for k, v in stat_dict.items():
                self.log_metric("statistics", k, v)
    
    def train(self) -> None:
        self.train_range = trange(self.step + 1, self.cfg.autoencoder.training.training_iters + 1)
        self.train_loader_iter = iter([])
        self._setup_train_data_generator()

        if not hasattr(self, "encodings_mean"):
            self.encodings_mean, self.encodings_std = self.get_encodings_statistics()

        self.get_latent_statistics()
        
        self.ddp_encoder.train()
        self.encoder.text_encoder.eval()
        self.ddp_decoder.train()

        for step in self.train_range:
            self.step = step
            
            batch = next(self.train_loader_iter, None)
            if batch is None:
                self._load_next_shard()
                batch = next(self.train_loader_iter, None)

            total_loss, loss_dict = self.calc_loss(batch)
            stat_dict = self.optimizer_step(total_loss)

            if self.step % self.cfg.autoencoder.logging.log_freq == 0:
                if dist.is_initialized() and dist.get_rank() == 0:
                    self.log_data(total_loss, loss_dict, stat_dict, is_train=True)   
            
            self.train_range.set_description(f"total_loss: {total_loss.item():0.3f}")
            
            if self.step % self.cfg.autoencoder.logging.save_freq == 0:
                self.latent_mean, self.latent_std = self.get_latent_statistics()
                self.save_checkpoint()

            if self.step % self.cfg.autoencoder.logging.eval_freq == 0:
                self.validate()
                torch.cuda.empty_cache()

        self.latent_mean, self.latent_std = self.get_latent_statistics()
        self.save_checkpoint()

        if dist.is_initialized() and dist.get_rank() == 0:
            wandb.finish()   

    @torch.no_grad()
    def validate(self) -> None:
        self._setup_valid_data_generator()
        self.ddp_encoder.eval()
        self.ddp_decoder.eval()
        

        total_loss = torch.Tensor([0.0])
        valid_dict: Dict[str, torch.Tensor] = dict()
        valid_count = torch.Tensor([0.0])

        for batch in self.valid_loader:
            batch_size = batch["input_ids"].shape[0]
            batch_loss, loss_dict = self.calc_loss(batch)
            
            for k in loss_dict:
                if k in valid_dict:
                    valid_dict[k] += loss_dict[k] * batch_size
                else:
                    valid_dict[k] = torch.Tensor([loss_dict[k] * batch_size])
            valid_count += batch_size

            total_loss += batch_loss.item() * batch_size

        
        if self.cfg.ddp.enabled:
            valid_count = reduce_tensor(valid_count.cuda())
            total_loss = reduce_tensor(total_loss.cuda())
        
            for k in valid_dict:
                valid_dict[k] = reduce_tensor(valid_dict[k].cuda()) / valid_count
        else:
            total_loss = total_loss / valid_count

            for k in valid_dict:
                valid_dict[k] = valid_dict[k] / valid_count
        
        self.log_data(total_loss, valid_dict, is_train=False)

        self.ddp_decoder.train()
        self.ddp_encoder.train()
        self.encoder.text_encoder.eval()

    def get_latent(self, batch, bert_output_masking: bool = False):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            # get bert hidden state
            with torch.no_grad():
                bert_hidden_state = self.encoder.text_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                ).last_hidden_state

                bert_hidden_state = self.normalize_encodings(bert_hidden_state)

            # masking bert hidden state
            if bert_output_masking:
                corrupted_bert_hidden_state = apply_corruption(
                    encodings=bert_hidden_state.detach().clone(),
                    mask=batch["mask"],
                    alpha=batch["alpha"],
                    noise=batch["noise"]
                )
                attention_mask_after_corruption = batch["corrupted_attention_mask"]


                # get latents
                encoder_latents = self.ddp_encoder(
                    token_ids=input_ids,
                    mask_tokens=attention_mask_after_corruption,
                    token_embeddings=corrupted_bert_hidden_state
                )
            else:
                encoder_latents = self.ddp_encoder(
                    token_ids=input_ids,
                    mask_tokens=attention_mask,
                    token_embeddings=bert_hidden_state
                )
        return encoder_latents, bert_hidden_state
            

    def calc_loss(self, batch) -> Tuple[Dict[str, torch.Tensor]]:
        batch = batch.to(self.device)
        if self.cfg.suffix == "v1.0":
            latents, bert_hidden_state = self.get_latent(batch, bert_output_masking=False)
        elif self.cfg.suffix == "v2.0":
            latents, bert_hidden_state = self.get_latent(batch, bert_output_masking=False)
        else:
            latents, bert_hidden_state = self.get_latent(batch, bert_output_masking=True)

        # Corrupt latents
        if self.cfg.suffix == "final":
            p = self.cfg.encoder.augmentation.latent_masking.probability
            latents = latents * (torch.rand_like(latents) > p)
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits, hidden_state_of_decoder = self.ddp_decoder(
                encoder_latents=latents, 
                return_last_hidden_state=True,
            )
        
        # Compute loss
        seq_len = batch["input_ids"].shape[1]
        ce_loss = cross_entropy_loss(
            input=logits[:, :seq_len],
            target=batch["input_ids"],
            mask=batch["attention_mask"],
        )
        mse_loss = mse_loss_function(
            input=hidden_state_of_decoder[:, :seq_len],
            target=bert_hidden_state.detach().clone(),
            mask=batch["attention_mask"],
        )
        variation_loss = total_variation_loss(latents)
        if self.cfg.suffix == "v1.0":
            total_loss = ce_loss
        else:
            total_loss = ce_loss + mse_loss
        
        # Logging
        stat_dict = {}
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16), torch.no_grad():
            stat_dict["ce_loss"] = ce_loss.detach().item()
            stat_dict["mse_loss"] = mse_loss.detach().item()

            acc = accuracy(
                logits=logits[:, :seq_len],
                target=batch["input_ids"],
                mask=batch["attention_mask"]
            )
            stat_dict["accuracy"] = acc.detach().item()

            stat_dict["variation_loss"] = variation_loss.detach().item()

        return total_loss, stat_dict

    @torch.no_grad()
    def reconstruction(self, output_file):
        self.set_valid_data_generator()
        self.encoder.eval()
        self.decoder.eval()
        
        result = []
        num_latent = self.cfg.encoder.latent.num_latents
        
        for batch in self.valid_loader:
            batch = batch.to(self.device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                encoder_latents = self.ddp_encoder(token_ids=batch["input_ids"], mask_tokens=batch["attention_mask"])
                latents = encoder_latents[:, :num_latent]
                decoder_masked_ids = torch.ones_like(batch["input_ids"], device=self.device) * self.tokenizer.mask_token_id
                decoder_masked_input = self.encoder.text_encoder.embeddings(decoder_masked_ids).detach().clone()
                logits = self.decoder(latents, masked_input=decoder_masked_input)
                pred_tokens = torch.argmax(logits, dim=-1)
            
            batch_size = batch["input_ids"].shape[0]
            seq_len = batch["input_ids"].shape[1]
            
            ce_loss = cross_entropy(
                input=logits.view(-1, logits.shape[-1]),
                target=batch["input_ids"].view(-1),
                reduce=False,
            )
            ce_loss = ce_loss.reshape((batch_size, seq_len))
            
            accuracy = (pred_tokens == batch["input_ids"]) * 1.
            
            target_text = self.tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=False)
            pred_text = self.tokenizer.batch_decode(pred_tokens, skip_special_tokens=False)
            
            for ind in range(batch_size):
                result.append(
                    {
                        "target": target_text[ind],
                        "prediction": pred_text[ind],
                        "target_tokens": to_str(batch["input_ids"][ind].tolist()),
                        "prediction_tokens": to_str(pred_tokens[ind].tolist()),
                        "loss": ce_loss[ind].mean().item(),
                        "accuracy": accuracy[ind].mean().item(),
                    }
                )
            break
        
        loss = np.mean([r["loss"] for r in result])
        accuracy = np.mean([r["accuracy"] for r in result])
        self.logger.info(f"Reconstruction loss: {loss:0.3f}")
        self.logger.info(f"Reconstruction accuracy: {accuracy:0.3f}")
        
        json.dump(result, open(output_file, "w"), indent=4)

    @torch.no_grad()
    def get_latent_statistics(self,):
        self._setup_valid_data_generator()
        self.ddp_encoder.eval()

        num_latents = self.cfg.encoder.latent.num_latents
        latent_sum = torch.zeros((num_latents, self.cfg.encoder.latent.dim), device=self.device)
        latent_sum_of_squares = torch.zeros((num_latents, self.cfg.encoder.latent.dim), device=self.device)
        latent_count = torch.Tensor([0.0]).to(self.device)
        
        for batch in self.valid_loader:
            batch = batch.to(self.device)
            
            latents, _ = self.get_latent(batch, bert_output_masking=False)
            latent_sum += latents.sum(dim=0)    
            latent_sum_of_squares += (latents ** 2).sum(dim=0)
            latent_count += latents.shape[0]

        if self.cfg.ddp.enabled:
            latent_count = reduce_tensor(latent_count)
            latent_sum = reduce_tensor(latent_sum)
            latent_sum_of_squares = reduce_tensor(latent_sum_of_squares)
        
        latent_mean = latent_sum / latent_count
        latent_sqr = torch.clip((latent_sum_of_squares / latent_count - latent_mean ** 2), min=1e-4)
        latent_std = torch.sqrt(latent_sqr)

        return latent_mean, latent_std
    
    def normalize_latent(self, latent):
        return (latent - self.latent_mean) / self.latent_std
    
    def denormalize_latent(self, latent):
        return latent * self.latent_std + self.latent_mean
    
    @torch.no_grad()
    def get_encodings_statistics(self,):
        self._setup_valid_data_generator()
        self.encoder.eval()
        
        encodings_sum = torch.zeros(self.cfg.encoder.embedding.dim, device=self.device)
        encodings_sum_of_squares = torch.zeros(self.cfg.encoder.embedding.dim, device=self.device)
        encodings_count = torch.Tensor([0.0]).to(self.device)
        
        for batch in tqdm(self.valid_loader):
            batch = batch.to(self.device)
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16), torch.no_grad():
                bert_hidden_state = self.encoder.text_encoder(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]
                ).last_hidden_state

                bert_hidden_state = bert_hidden_state.reshape(-1, bert_hidden_state.shape[-1])
                mask = batch["attention_mask"].reshape(-1).bool()
                bert_hidden_state = bert_hidden_state[mask]

            encodings_sum += bert_hidden_state.sum(dim=0)    
            encodings_sum_of_squares += (bert_hidden_state ** 2).sum(dim=0)
            encodings_count += bert_hidden_state.shape[0]
        if self.cfg.ddp.enabled:
            encodings_count = reduce_tensor(encodings_count)
            encodings_sum = reduce_tensor(encodings_sum)
            encodings_sum_of_squares = reduce_tensor(encodings_sum_of_squares)

        encodings_mean = encodings_sum / encodings_count
        encodings_sqr = (encodings_sum_of_squares / encodings_count - encodings_mean ** 2)
        encodings_std = torch.sqrt(torch.clip(encodings_sqr, min=1e-4))
        return encodings_mean, encodings_std
    
    def normalize_encodings(self, encodings):
        return (encodings - self.encodings_mean) / self.encodings_std
    
    def denormalize_encodings(self, encodings):
        return encodings * self.encodings_std + self.encodings_mean
         
        
        
        
        
