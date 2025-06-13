"""
Singular Value Fine-tuning (SVF) implementation.
Provides parameter-efficient adaptation by tuning only singular values of weight matrices.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import numpy as np
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

# Optional wandb import - handled at runtime
HAS_WANDB = False
wandb = None

def _init_wandb():
    """Initialize wandb if available."""
    global HAS_WANDB, wandb
    if wandb is None:
        try:
            import wandb as _wandb
            wandb = _wandb
            HAS_WANDB = True
        except ImportError:
            HAS_WANDB = False
            print("Warning: wandb not available. Logging will be disabled.")
    return HAS_WANDB

@dataclass
class SVFConfig:
    """Configuration for Singular Value Fine-tuning."""
    rank: int = 16  # SVF rank
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_epochs: int = 10
    warmup_steps: int = 100
    gradient_clip_norm: float = 1.0
    save_steps: int = 500
    eval_steps: int = 100
    logging_steps: int = 50
    
    # Regularization
    svf_alpha: float = 1.0  # SVF loss weight
    orthogonal_reg: float = 0.1  # Orthogonality regularization
    sparsity_reg: float = 0.05  # Sparsity regularization on singular values
    
    # Training stability
    adaptive_rank: bool = True  # Dynamically adjust rank
    min_rank: int = 4
    max_rank: int = 64
    rank_threshold: float = 0.01  # Threshold for singular value magnitude
    
    # Optimization
    optimizer: str = "adamw"  # adamw, adam, sgd
    scheduler: str = "linear"  # linear, cosine, constant
    
    # Logging
    use_wandb: bool = True
    project_name: str = "transformer-squared-svf"
    experiment_name: str = "svf_training"


class SVFOptimizer:
    """Specialized optimizer for SVF parameters."""
    
    def __init__(self, svf_parameters: List[nn.Parameter], config: SVFConfig):
        self.config = config
        self.svf_parameters = svf_parameters
        
        if config.optimizer == "adamw":
            self.optimizer = optim.AdamW(
                svf_parameters,
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        elif config.optimizer == "adam":
            self.optimizer = optim.Adam(
                svf_parameters,
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        elif config.optimizer == "sgd":
            self.optimizer = optim.SGD(
                svf_parameters,
                lr=config.learning_rate,
                momentum=0.9,
                weight_decay=config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {config.optimizer}")
    
    def step(self):
        """Perform optimization step."""
        self.optimizer.step()
    
    def zero_grad(self):
        """Zero gradients."""
        self.optimizer.zero_grad()
    
    def state_dict(self):
        """Get optimizer state."""
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load optimizer state."""
        self.optimizer.load_state_dict(state_dict)


class SVFRegularizer:
    """Regularization utilities for SVF training."""
    
    @staticmethod
    def orthogonal_loss(U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        Compute orthogonality regularization loss.
        
        Args:
            U: Left singular vectors [output_dim, rank]
            V: Right singular vectors [rank, input_dim]
            
        Returns:
            Orthogonality loss
        """
        # U should have orthogonal columns
        U_orth_loss = torch.norm(U.T @ U - torch.eye(U.shape[1], device=U.device)) ** 2
        
        # V should have orthogonal rows
        V_orth_loss = torch.norm(V @ V.T - torch.eye(V.shape[0], device=V.device)) ** 2
        
        return U_orth_loss + V_orth_loss
    
    @staticmethod
    def sparsity_loss(S: torch.Tensor, threshold: float = 0.01) -> torch.Tensor:
        """
        Compute sparsity regularization on singular values.
        
        Args:
            S: Singular values [rank]
            threshold: Sparsity threshold
            
        Returns:
            Sparsity loss
        """
        # Encourage small singular values to become zero
        return torch.sum(torch.relu(threshold - torch.abs(S)))
    
    @staticmethod
    def adaptive_rank_pruning(S: torch.Tensor, threshold: float = 0.01) -> Tuple[torch.Tensor, int]:
        """
        Adaptively prune singular values below threshold.
        
        Args:
            S: Singular values [rank]
            threshold: Pruning threshold
            
        Returns:
            Tuple of (pruned_S, effective_rank)
        """
        mask = torch.abs(S) > threshold
        effective_rank = mask.sum().item()
        pruned_S = S * mask.float()
        return pruned_S, effective_rank


class SVFMetrics:
    """Metrics tracking for SVF training."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.total_loss = 0.0
        self.svf_loss = 0.0
        self.orthogonal_loss = 0.0
        self.sparsity_loss = 0.0
        self.effective_ranks = []
        self.singular_value_stats = []
        self.num_samples = 0
    
    def update(self, losses: Dict[str, float], effective_ranks: List[int], singular_values: List[torch.Tensor]):
        """Update metrics."""
        self.total_loss += losses.get('total', 0.0)
        self.svf_loss += losses.get('svf', 0.0)
        self.orthogonal_loss += losses.get('orthogonal', 0.0)
        self.sparsity_loss += losses.get('sparsity', 0.0)
        self.effective_ranks.extend(effective_ranks)
        
        # Compute singular value statistics
        for sv in singular_values:
            stats = {
                'mean': sv.mean().item(),
                'std': sv.std().item(),
                'max': sv.max().item(),
                'min': sv.min().item()
            }
            self.singular_value_stats.append(stats)
        
        self.num_samples += 1
    
    def get_averages(self) -> Dict[str, float]:
        """Get average metrics."""
        if self.num_samples == 0:
            return {}
        
        metrics = {
            'avg_total_loss': self.total_loss / self.num_samples,
            'avg_svf_loss': self.svf_loss / self.num_samples,
            'avg_orthogonal_loss': self.orthogonal_loss / self.num_samples,
            'avg_sparsity_loss': self.sparsity_loss / self.num_samples,
            'avg_effective_rank': np.mean(self.effective_ranks) if self.effective_ranks else 0,
            'std_effective_rank': np.std(self.effective_ranks) if self.effective_ranks else 0,
        }
        
        if self.singular_value_stats:
            sv_means = [s['mean'] for s in self.singular_value_stats]
            sv_stds = [s['std'] for s in self.singular_value_stats]
            metrics.update({
                'avg_sv_mean': np.mean(sv_means),
                'avg_sv_std': np.mean(sv_stds),
            })
        
        return metrics


class SVFTrainer:
    """
    Trainer for Singular Value Fine-tuning.
    
    Implements efficient training of SVF parameters with regularization
    and adaptive rank management.
    """
    
    def __init__(self, model, config: SVFConfig):
        self.model = model
        self.config = config
        
        # Collect SVF parameters
        self.svf_parameters = self._collect_svf_parameters()
        
        # Initialize optimizer
        self.optimizer = SVFOptimizer(self.svf_parameters, config)
        
        # Initialize scheduler
        self.scheduler = None
        
        # Metrics tracking
        self.metrics = SVFMetrics()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        
        # Initialize wandb if enabled and available
        if config.use_wandb:
            if _init_wandb():
                wandb.init(
                    project=config.project_name,
                    name=config.experiment_name,
                    config=config.__dict__
                )
            else:
                print("Warning: wandb requested but not available. Skipping wandb initialization.")
    
    def _collect_svf_parameters(self) -> List[nn.Parameter]:
        """Collect all SVF parameters from the model."""
        svf_parameters = []
        
        for name, module in self.model.named_modules():
            if hasattr(module, 'U') and hasattr(module, 'S') and hasattr(module, 'V'):
                # This is an SVF layer
                svf_parameters.extend([module.U, module.S, module.V])
        
        print(f"Found {len(svf_parameters)} SVF parameters")
        return svf_parameters
    
    def _setup_scheduler(self, num_training_steps: int):
        """Setup learning rate scheduler."""
        if self.config.scheduler == "linear":
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer.optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=num_training_steps
            )
        elif self.config.scheduler == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer.optimizer,
                T_max=num_training_steps
            )
        # No scheduler for "constant"
    
    def _compute_svf_loss(self, outputs, labels) -> Dict[str, torch.Tensor]:
        """Compute SVF-specific losses."""
        # Base loss (cross-entropy)
        base_loss = outputs.loss if hasattr(outputs, 'loss') else 0.0
        
        # Regularization losses
        orthogonal_loss = 0.0
        sparsity_loss = 0.0
        effective_ranks = []
        singular_values = []
        
        for name, module in self.model.named_modules():
            if hasattr(module, 'U') and hasattr(module, 'S') and hasattr(module, 'V'):
                # Orthogonality regularization
                orth_loss = SVFRegularizer.orthogonal_loss(module.U, module.V)
                orthogonal_loss += orth_loss
                
                # Sparsity regularization
                sparse_loss = SVFRegularizer.sparsity_loss(module.S, self.config.rank_threshold)
                sparsity_loss += sparse_loss
                
                # Track effective rank
                _, eff_rank = SVFRegularizer.adaptive_rank_pruning(module.S, self.config.rank_threshold)
                effective_ranks.append(eff_rank)
                singular_values.append(module.S.detach().clone())
        
        # Total loss
        total_loss = (self.config.svf_alpha * base_loss + 
                     self.config.orthogonal_reg * orthogonal_loss +
                     self.config.sparsity_reg * sparsity_loss)
        
        losses = {
            'total': total_loss,
            'svf': base_loss,
            'orthogonal': orthogonal_loss,
            'sparsity': sparsity_loss
        }
        
        return losses, effective_ranks, singular_values
    
    def train_step(self, batch) -> Dict[str, float]:
        """Perform a single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        input_ids = batch['input_ids']
        attention_mask = batch.get('attention_mask', None)
        labels = batch.get('labels', input_ids)
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_two_pass=False  # Single pass during training
        )
        
        # Compute losses
        losses, effective_ranks, singular_values = self._compute_svf_loss(outputs, labels)
        
        # Backward pass
        losses['total'].backward()
        
        # Gradient clipping
        if self.config.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.svf_parameters, self.config.gradient_clip_norm)
        
        # Optimization step
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()
        
        # Convert losses to float for logging
        loss_dict = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}
        
        # Update metrics
        self.metrics.update(loss_dict, effective_ranks, singular_values)
        
        self.global_step += 1
        
        return loss_dict
    
    def evaluate(self, eval_dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate the model."""
        self.model.eval()
        eval_metrics = SVFMetrics()
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                input_ids = batch['input_ids']
                attention_mask = batch.get('attention_mask', None)
                labels = batch.get('labels', input_ids)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    use_two_pass=True  # Use two-pass during evaluation
                )
                
                losses, effective_ranks, singular_values = self._compute_svf_loss(outputs, labels)
                loss_dict = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}
                eval_metrics.update(loss_dict, effective_ranks, singular_values)
        
        return eval_metrics.get_averages()
    
    def train(self, train_dataloader: DataLoader, eval_dataloader: Optional[DataLoader] = None):
        """
        Train the SVF parameters.
        
        Args:
            train_dataloader: Training data loader
            eval_dataloader: Optional evaluation data loader
        """
        num_training_steps = len(train_dataloader) * self.config.max_epochs
        self._setup_scheduler(num_training_steps)
        
        print(f"Starting SVF training for {self.config.max_epochs} epochs")
        print(f"Total training steps: {num_training_steps}")
        
        for epoch in range(self.config.max_epochs):
            self.epoch = epoch
            self.metrics.reset()
            
            # Training loop
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{self.config.max_epochs}")
            for step, batch in enumerate(progress_bar):
                loss_dict = self.train_step(batch)
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    avg_metrics = self.metrics.get_averages()
                    progress_bar.set_postfix({
                        'loss': f"{avg_metrics.get('avg_total_loss', 0):.4f}",
                        'rank': f"{avg_metrics.get('avg_effective_rank', 0):.1f}"
                    })
                    
                    if self.config.use_wandb and HAS_WANDB and wandb is not None:
                        wandb.log({
                            'train/loss': avg_metrics.get('avg_total_loss', 0),
                            'train/svf_loss': avg_metrics.get('avg_svf_loss', 0),
                            'train/orthogonal_loss': avg_metrics.get('avg_orthogonal_loss', 0),
                            'train/sparsity_loss': avg_metrics.get('avg_sparsity_loss', 0),
                            'train/effective_rank': avg_metrics.get('avg_effective_rank', 0),
                            'train/lr': self.optimizer.optimizer.param_groups[0]['lr'],
                            'step': self.global_step
                        })
                
                # Evaluation
                if eval_dataloader and self.global_step % self.config.eval_steps == 0:
                    eval_metrics = self.evaluate(eval_dataloader)
                    print(f"\nEvaluation at step {self.global_step}:")
                    for key, value in eval_metrics.items():
                        print(f"  {key}: {value:.4f}")
                    
                    if self.config.use_wandb and HAS_WANDB and wandb is not None:
                        wandb.log({f'eval/{k}': v for k, v in eval_metrics.items()}, step=self.global_step)
                
                # Save checkpoint
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint(f"checkpoint-{self.global_step}")
        
        print("SVF training completed!")
        
        # Final evaluation
        if eval_dataloader:
            final_metrics = self.evaluate(eval_dataloader)
            print("\nFinal evaluation results:")
            for key, value in final_metrics.items():
                print(f"  {key}: {value:.4f}")
    
    def save_checkpoint(self, checkpoint_dir: str):
        """Save training checkpoint."""
        import os
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'optimizer_state': self.optimizer.state_dict(),
            'config': self.config.__dict__
        }
        
        if self.scheduler:
            checkpoint['scheduler_state'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, os.path.join(checkpoint_dir, "training_state.pt"))
        
        # Save model adapted weights
        self.model.save_adapted_model(checkpoint_dir)
        
        print(f"Checkpoint saved to {checkpoint_dir}")
    
    def load_checkpoint(self, checkpoint_dir: str):
        """Load training checkpoint."""
        import os
        
        checkpoint_path = os.path.join(checkpoint_dir, "training_state.pt")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            
            self.global_step = checkpoint['global_step']
            self.epoch = checkpoint['epoch']
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            
            if self.scheduler and 'scheduler_state' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state'])
            
            print(f"Training state loaded from {checkpoint_dir}")
        
        # Load model weights
        self.model.load_adapted_model(checkpoint_dir) 