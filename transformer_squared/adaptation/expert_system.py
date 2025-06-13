"""
Expert System for Transformer-Squared framework.
Implements dynamic expert mixing and reinforcement learning for expert training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
import json

@dataclass
class ExpertConfig:
    """Configuration for Expert System."""
    num_experts: int = 8
    expert_dim: int = 64
    hidden_size: int = 2048  # Model hidden size
    gating_hidden_dim: int = 256
    
    # Expert specialization
    expert_specialization: bool = True
    specialization_strength: float = 0.1
    diversity_loss_weight: float = 0.01
    
    # Reinforcement learning
    use_rl_training: bool = True
    reward_window: int = 100
    baseline_momentum: float = 0.9
    policy_lr: float = 1e-4
    value_lr: float = 1e-3
    
    # Expert memory
    use_expert_memory: bool = True
    memory_size: int = 1000
    memory_update_freq: int = 10
    
    # Task identification
    task_embedding_dim: int = 64
    num_task_clusters: int = 20
    cluster_update_freq: int = 50


class ExpertMemory:
    """Memory system for storing expert usage patterns and performance."""
    
    def __init__(self, config: ExpertConfig):
        self.config = config
        self.memory = defaultdict(list)  # expert_id -> list of (task_embedding, performance)
        self.task_expert_mapping = {}  # task_hash -> best_expert_id
        
    def store_experience(self, expert_id: int, task_embedding: torch.Tensor, performance: float):
        """Store expert performance for a given task."""
        if len(self.memory[expert_id]) >= self.config.memory_size:
            self.memory[expert_id].pop(0)  # Remove oldest
        
        self.memory[expert_id].append({
            'task_embedding': task_embedding.detach().cpu(),
            'performance': performance,
            'timestamp': len(self.memory[expert_id])
        })
    
    def get_expert_recommendations(self, task_embedding: torch.Tensor, top_k: int = 3) -> List[int]:
        """Get recommended experts for a given task."""
        similarities = {}
        
        for expert_id, experiences in self.memory.items():
            if not experiences:
                continue
                
            # Compute similarity to stored experiences
            task_similarities = []
            performances = []
            
            for exp in experiences[-50:]:  # Use recent experiences
                sim = F.cosine_similarity(
                    task_embedding.cpu().unsqueeze(0),
                    exp['task_embedding'].unsqueeze(0)
                ).item()
                task_similarities.append(sim)
                performances.append(exp['performance'])
            
            if task_similarities:
                # Weight by similarity and performance
                weighted_score = np.average(performances, weights=task_similarities)
                similarities[expert_id] = weighted_score
        
        # Return top-k experts
        sorted_experts = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        return [expert_id for expert_id, _ in sorted_experts[:top_k]]
    
    def get_expert_stats(self) -> Dict[int, Dict[str, float]]:
        """Get statistics for each expert."""
        stats = {}
        
        for expert_id, experiences in self.memory.items():
            if not experiences:
                continue
                
            performances = [exp['performance'] for exp in experiences]
            stats[expert_id] = {
                'avg_performance': np.mean(performances),
                'std_performance': np.std(performances),
                'num_experiences': len(experiences),
                'recent_performance': np.mean(performances[-10:]) if len(performances) >= 10 else np.mean(performances)
            }
        
        return stats


class TaskClusterManager:
    """Manages task clustering for expert specialization."""
    
    def __init__(self, config: ExpertConfig):
        self.config = config
        self.cluster_centers = nn.Parameter(
            torch.randn(config.num_task_clusters, config.task_embedding_dim)
        )
        self.cluster_assignments = {}
        self.cluster_expert_affinity = torch.zeros(config.num_task_clusters, config.num_experts)
        
    def assign_task_to_cluster(self, task_embedding: torch.Tensor) -> int:
        """Assign task to nearest cluster."""
        similarities = F.cosine_similarity(
            task_embedding.unsqueeze(0),
            self.cluster_centers
        )
        cluster_id = similarities.argmax().item()
        return cluster_id
    
    def update_cluster_expert_affinity(self, cluster_id: int, expert_weights: torch.Tensor):
        """Update affinity between clusters and experts."""
        # Exponential moving average
        alpha = 0.1
        self.cluster_expert_affinity[cluster_id] = (
            (1 - alpha) * self.cluster_expert_affinity[cluster_id] + 
            alpha * expert_weights.detach().cpu()
        )
    
    def get_cluster_expert_preferences(self, cluster_id: int) -> torch.Tensor:
        """Get expert preferences for a cluster."""
        return self.cluster_expert_affinity[cluster_id]


class ExpertGatingNetwork(nn.Module):
    """Gating network for expert selection."""
    
    def __init__(self, config: ExpertConfig):
        super().__init__()
        self.config = config
        
        # Main gating network
        self.gate = nn.Sequential(
            nn.Linear(config.task_embedding_dim, config.gating_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.gating_hidden_dim, config.gating_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.gating_hidden_dim // 2, config.num_experts)
        )
        
        # Value network for RL
        if config.use_rl_training:
            self.value_network = nn.Sequential(
                nn.Linear(config.task_embedding_dim, config.gating_hidden_dim),
                nn.ReLU(),
                nn.Linear(config.gating_hidden_dim, 1)
            )
        
        # Temperature for exploration
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, task_embedding: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through gating network.
        
        Args:
            task_embedding: Task embedding [batch_size, task_embedding_dim]
            training: Whether in training mode
            
        Returns:
            expert_weights: Expert selection probabilities [batch_size, num_experts]
            logits: Raw logits before softmax [batch_size, num_experts]
        """
        logits = self.gate(task_embedding)
        
        if training:
            # Add exploration noise during training
            temperature = torch.clamp(self.temperature, min=0.1, max=2.0)
            expert_weights = F.softmax(logits / temperature, dim=-1)
        else:
            # Greedy selection during inference
            expert_weights = F.softmax(logits, dim=-1)
        
        return expert_weights, logits
    
    def get_value(self, task_embedding: torch.Tensor) -> torch.Tensor:
        """Get value estimate for RL training."""
        if hasattr(self, 'value_network'):
            return self.value_network(task_embedding)
        return torch.zeros(task_embedding.shape[0], 1, device=task_embedding.device)


class ExpertVectors(nn.Module):
    """Expert vector storage and combination."""
    
    def __init__(self, config: ExpertConfig):
        super().__init__()
        self.config = config
        
        # Expert vectors
        self.expert_vectors = nn.Parameter(
            torch.randn(config.num_experts, config.expert_dim) * 0.01
        )
        
        # Projection layers
        self.input_projection = nn.Linear(config.hidden_size, config.expert_dim)
        self.output_projection = nn.Linear(config.expert_dim, config.hidden_size)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        
    def forward(self, hidden_states: torch.Tensor, expert_weights: torch.Tensor) -> torch.Tensor:
        """
        Apply expert mixing to hidden states.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            expert_weights: Expert selection weights [batch_size, num_experts]
            
        Returns:
            Adapted hidden states [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Project to expert dimension
        projected = self.input_projection(hidden_states)  # [batch, seq_len, expert_dim]
        
        # Compute expert mixing
        mixed_experts = torch.einsum('be,ed->bd', expert_weights, self.expert_vectors)  # [batch, expert_dim]
        mixed_experts = mixed_experts.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq_len, expert_dim]
        
        # Apply expert influence
        adapted = projected + mixed_experts
        
        # Project back to hidden size
        output = self.output_projection(adapted)  # [batch, seq_len, hidden_size]
        
        # Residual connection and layer norm
        return self.layer_norm(hidden_states + output)
    
    def get_expert_diversity_loss(self) -> torch.Tensor:
        """Compute diversity loss to encourage expert specialization."""
        # Compute pairwise similarities between experts
        normalized_experts = F.normalize(self.expert_vectors, p=2, dim=1)
        similarity_matrix = torch.mm(normalized_experts, normalized_experts.t())
        
        # Remove diagonal (self-similarity)
        mask = torch.eye(self.config.num_experts, device=self.expert_vectors.device)
        similarity_matrix = similarity_matrix * (1 - mask)
        
        # Encourage low similarity (high diversity)
        diversity_loss = torch.mean(similarity_matrix ** 2)
        return diversity_loss


class ReinforcementLearningTrainer:
    """RL trainer for expert system."""
    
    def __init__(self, expert_system, config: ExpertConfig):
        self.expert_system = expert_system
        self.config = config
        
        # Experience buffer
        self.experiences = []
        self.baseline = 0.0
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(
            expert_system.gating_network.gate.parameters(),
            lr=config.policy_lr
        )
        
        if config.use_rl_training:
            self.value_optimizer = torch.optim.Adam(
                expert_system.gating_network.value_network.parameters(),
                lr=config.value_lr
            )
    
    def store_experience(self, task_embedding: torch.Tensor, action_logits: torch.Tensor, 
                        expert_weights: torch.Tensor, reward: float):
        """Store experience for RL training."""
        experience = {
            'task_embedding': task_embedding.detach(),
            'action_logits': action_logits.detach(),
            'expert_weights': expert_weights.detach(),
            'reward': reward
        }
        
        if len(self.experiences) >= self.config.reward_window:
            self.experiences.pop(0)
        
        self.experiences.append(experience)
    
    def update_policy(self):
        """Update policy using REINFORCE with baseline."""
        if len(self.experiences) < 10:
            return
        
        # Compute returns and advantages
        rewards = [exp['reward'] for exp in self.experiences]
        self.baseline = self.config.baseline_momentum * self.baseline + (1 - self.config.baseline_momentum) * np.mean(rewards)
        
        policy_loss = 0.0
        value_loss = 0.0
        
        for exp in self.experiences[-32:]:  # Use recent experiences
            task_embedding = exp['task_embedding']
            action_logits = exp['action_logits']
            expert_weights = exp['expert_weights']
            reward = exp['reward']
            
            # Compute advantage
            advantage = reward - self.baseline
            
            # Policy loss (REINFORCE)
            log_probs = F.log_softmax(action_logits, dim=-1)
            selected_log_probs = torch.sum(log_probs * expert_weights, dim=-1)
            policy_loss -= torch.mean(selected_log_probs * advantage)
            
            # Value loss
            if self.config.use_rl_training:
                value_pred = self.expert_system.gating_network.get_value(task_embedding.unsqueeze(0))
                value_target = torch.tensor([[reward]], device=value_pred.device, dtype=value_pred.dtype)
                value_loss += F.mse_loss(value_pred, value_target)
        
        # Update policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.expert_system.gating_network.gate.parameters(), 1.0)
        self.policy_optimizer.step()
        
        # Update value function
        if self.config.use_rl_training and value_loss > 0:
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.expert_system.gating_network.value_network.parameters(), 1.0)
            self.value_optimizer.step()


class ExpertSystem(nn.Module):
    """
    Complete Expert System for Transformer-Squared framework.
    
    Manages expert vectors, gating network, and reinforcement learning training.
    """
    
    def __init__(self, config: ExpertConfig):
        super().__init__()
        self.config = config
        
        # Core components
        self.gating_network = ExpertGatingNetwork(config)
        self.expert_vectors = ExpertVectors(config)
        
        # Memory and clustering
        if config.use_expert_memory:
            self.memory = ExpertMemory(config)
        
        self.cluster_manager = TaskClusterManager(config)
        
        # RL trainer
        if config.use_rl_training:
            self.rl_trainer = ReinforcementLearningTrainer(self, config)
        
        # Training state
        self.training_step = 0
        self.expert_usage_stats = defaultdict(int)
        
    def forward(self, hidden_states: torch.Tensor, task_embedding: torch.Tensor, 
                return_expert_info: bool = False) -> torch.Tensor:
        """
        Apply expert system to hidden states.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            task_embedding: Task embedding [batch_size, task_embedding_dim]
            return_expert_info: Whether to return expert selection info
            
        Returns:
            Adapted hidden states [batch_size, seq_len, hidden_size]
            If return_expert_info: (adapted_states, expert_info_dict)
        """
        # Get expert selection weights
        expert_weights, action_logits = self.gating_network(task_embedding, training=self.training)
        
        # Apply expert mixing
        adapted_states = self.expert_vectors(hidden_states, expert_weights)
        
        # Update usage statistics
        if self.training:
            selected_experts = expert_weights.argmax(dim=-1)
            for expert_id in selected_experts:
                self.expert_usage_stats[expert_id.item()] += 1
        
        # Store cluster information
        if self.training and self.training_step % self.config.cluster_update_freq == 0:
            for i, task_emb in enumerate(task_embedding):
                cluster_id = self.cluster_manager.assign_task_to_cluster(task_emb)
                self.cluster_manager.update_cluster_expert_affinity(cluster_id, expert_weights[i])
        
        if return_expert_info:
            expert_info = {
                'expert_weights': expert_weights,
                'action_logits': action_logits,
                'selected_experts': expert_weights.argmax(dim=-1),
                'expert_entropy': -torch.sum(expert_weights * torch.log(expert_weights + 1e-8), dim=-1).mean()
            }
            return adapted_states, expert_info
        
        return adapted_states
    
    def update_from_reward(self, task_embedding: torch.Tensor, expert_info: Dict, reward: float):
        """Update expert system based on reward signal."""
        if self.config.use_rl_training and hasattr(self, 'rl_trainer'):
            self.rl_trainer.store_experience(
                task_embedding,
                expert_info['action_logits'],
                expert_info['expert_weights'],
                reward
            )
            
            # Periodic policy update
            if self.training_step % 50 == 0:
                self.rl_trainer.update_policy()
        
        # Update memory
        if self.config.use_expert_memory and hasattr(self, 'memory'):
            selected_expert = expert_info['selected_experts'].item()
            self.memory.store_experience(selected_expert, task_embedding.squeeze(), reward)
        
        self.training_step += 1
    
    def get_expert_recommendations(self, task_embedding: torch.Tensor) -> List[int]:
        """Get expert recommendations for a task."""
        if hasattr(self, 'memory'):
            return self.memory.get_expert_recommendations(task_embedding)
        return list(range(self.config.num_experts))
    
    def get_expert_statistics(self) -> Dict[str, Any]:
        """Get comprehensive expert statistics."""
        stats = {
            'usage_stats': dict(self.expert_usage_stats),
            'training_step': self.training_step,
            'num_experts': self.config.num_experts
        }
        
        if hasattr(self, 'memory'):
            stats['expert_memory_stats'] = self.memory.get_expert_stats()
        
        # Expert diversity
        diversity_loss = self.expert_vectors.get_expert_diversity_loss()
        stats['expert_diversity'] = diversity_loss.item()
        
        # Cluster information
        stats['cluster_expert_affinity'] = self.cluster_manager.cluster_expert_affinity.tolist()
        
        return stats
    
    def save_expert_state(self, save_path: str):
        """Save expert system state."""
        state = {
            'gating_network': self.gating_network.state_dict(),
            'expert_vectors': self.expert_vectors.state_dict(),
            'cluster_centers': self.cluster_manager.cluster_centers,
            'cluster_expert_affinity': self.cluster_manager.cluster_expert_affinity,
            'training_step': self.training_step,
            'usage_stats': dict(self.expert_usage_stats),
            'config': self.config.__dict__
        }
        
        if hasattr(self, 'memory'):
            # Save memory (limited to avoid large files)
            memory_data = {}
            for expert_id, experiences in self.memory.memory.items():
                memory_data[expert_id] = experiences[-100:]  # Keep recent experiences
            state['expert_memory'] = memory_data
        
        torch.save(state, save_path)
        print(f"Expert system state saved to {save_path}")
    
    def load_expert_state(self, save_path: str):
        """Load expert system state."""
        state = torch.load(save_path)
        
        self.gating_network.load_state_dict(state['gating_network'])
        self.expert_vectors.load_state_dict(state['expert_vectors'])
        self.cluster_manager.cluster_centers = state['cluster_centers']
        self.cluster_manager.cluster_expert_affinity = state['cluster_expert_affinity']
        self.training_step = state['training_step']
        self.expert_usage_stats = defaultdict(int, state['usage_stats'])
        
        if 'expert_memory' in state and hasattr(self, 'memory'):
            for expert_id, experiences in state['expert_memory'].items():
                self.memory.memory[int(expert_id)] = experiences
        
        print(f"Expert system state loaded from {save_path}") 