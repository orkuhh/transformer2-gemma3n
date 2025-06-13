"""
Cross-Entropy Method (CEM) Adapter for Transformer-Squared framework.
Implements CEM-based optimization for expert vector adaptation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import torch.nn.functional as F
from scipy.stats import multivariate_normal


@dataclass
class CEMConfig:
    """Configuration for CEM Adapter."""
    # CEM parameters
    population_size: int = 100
    elite_fraction: float = 0.2
    noise_std: float = 0.1
    max_iterations: int = 50
    convergence_threshold: float = 1e-4
    
    # Adaptation parameters
    adaptation_rate: float = 0.1
    momentum: float = 0.9
    temperature_schedule: str = "exponential"  # exponential, linear, constant
    initial_temperature: float = 1.0
    final_temperature: float = 0.1
    
    # Expert selection
    top_k_experts: int = 3
    diversity_weight: float = 0.1
    
    # Regularization
    l2_penalty: float = 0.01
    entropy_bonus: float = 0.05


class CEMDistribution:
    """Multivariate Gaussian distribution for CEM."""
    
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean
        self.std = std
        self.dim = mean.shape[-1]
        
    def sample(self, n_samples: int) -> torch.Tensor:
        """Sample from the distribution."""
        device = self.mean.device
        samples = torch.randn(n_samples, self.dim, device=device)
        return self.mean.unsqueeze(0) + samples * self.std.unsqueeze(0)
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log probability of samples."""
        diff = x - self.mean.unsqueeze(0)
        log_prob = -0.5 * torch.sum((diff / self.std.unsqueeze(0)) ** 2, dim=-1)
        log_prob -= 0.5 * self.dim * np.log(2 * np.pi)
        log_prob -= torch.sum(torch.log(self.std))
        return log_prob
    
    def update(self, elite_samples: torch.Tensor, momentum: float = 0.0):
        """Update distribution parameters with elite samples."""
        new_mean = torch.mean(elite_samples, dim=0)
        new_std = torch.std(elite_samples, dim=0) + 1e-6  # Add small epsilon for numerical stability
        
        if momentum > 0:
            self.mean = momentum * self.mean + (1 - momentum) * new_mean
            self.std = momentum * self.std + (1 - momentum) * new_std
        else:
            self.mean = new_mean
            self.std = new_std


class ObjectiveFunction:
    """Base class for CEM objective functions."""
    
    def __call__(self, samples: torch.Tensor, context: Dict[str, Any]) -> torch.Tensor:
        """
        Evaluate objective function.
        
        Args:
            samples: Candidate solutions [n_samples, dim]
            context: Additional context information
            
        Returns:
            Objective values [n_samples]
        """
        raise NotImplementedError


class ExpertSelectionObjective(ObjectiveFunction):
    """Objective function for expert selection optimization."""
    
    def __init__(self, expert_system, config: CEMConfig):
        self.expert_system = expert_system
        self.config = config
        
    def __call__(self, samples: torch.Tensor, context: Dict[str, Any]) -> torch.Tensor:
        """
        Evaluate expert selection samples.
        
        Args:
            samples: Expert weight samples [n_samples, num_experts]
            context: Contains task_embedding, hidden_states, etc.
            
        Returns:
            Objective values [n_samples]
        """
        n_samples = samples.shape[0]
        task_embedding = context['task_embedding']
        hidden_states = context.get('hidden_states')
        target_output = context.get('target_output')
        
        objectives = []
        
        for i in range(n_samples):
            expert_weights = F.softmax(samples[i], dim=-1)
            
            # Apply expert mixing
            if hidden_states is not None:
                adapted_output = self.expert_system.expert_vectors(
                    hidden_states.unsqueeze(0), 
                    expert_weights.unsqueeze(0)
                )
                
                # Compute task-specific objective
                if target_output is not None:
                    # Reconstruction loss
                    recon_loss = F.mse_loss(adapted_output, target_output.unsqueeze(0))
                    objective = -recon_loss.item()
                else:
                    # Use diversity and novelty as objectives
                    diversity = torch.sum(expert_weights * torch.log(expert_weights + 1e-8))
                    objective = diversity.item()
            else:
                # Pure expert selection objective
                diversity = -torch.sum(expert_weights * torch.log(expert_weights + 1e-8))
                
                # Expert quality from memory
                if hasattr(self.expert_system, 'memory'):
                    expert_scores = []
                    for j, weight in enumerate(expert_weights):
                        if j in self.expert_system.memory.memory:
                            experiences = self.expert_system.memory.memory[j]
                            if experiences:
                                recent_perf = np.mean([exp['performance'] for exp in experiences[-10:]])
                                expert_scores.append(weight * recent_perf)
                            else:
                                expert_scores.append(0.0)
                        else:
                            expert_scores.append(0.0)
                    
                    quality_score = sum(expert_scores)
                    objective = quality_score + self.config.diversity_weight * diversity.item()
                else:
                    objective = diversity.item()
            
            # Add regularization
            l2_penalty = torch.sum(expert_weights ** 2) * self.config.l2_penalty
            objective -= l2_penalty.item()
            
            objectives.append(objective)
        
        return torch.tensor(objectives, device=samples.device)


class ParameterOptimizationObjective(ObjectiveFunction):
    """Objective function for parameter optimization."""
    
    def __init__(self, model, config: CEMConfig):
        self.model = model
        self.config = config
        
    def __call__(self, samples: torch.Tensor, context: Dict[str, Any]) -> torch.Tensor:
        """
        Evaluate parameter samples.
        
        Args:
            samples: Parameter samples [n_samples, param_dim]
            context: Contains input data and targets
            
        Returns:
            Objective values [n_samples]
        """
        input_data = context['input_data']
        targets = context.get('targets')
        
        objectives = []
        
        # Store original parameters
        original_params = {}
        param_names = context.get('param_names', [])
        
        for name in param_names:
            param = self.model.get_parameter(name)
            if param is not None:
                original_params[name] = param.data.clone()
        
        for i, sample in enumerate(samples):
            # Apply parameter sample
            param_idx = 0
            for name in param_names:
                param = self.model.get_parameter(name)
                if param is not None:
                    param_size = param.numel()
                    param.data = sample[param_idx:param_idx + param_size].view(param.shape)
                    param_idx += param_size
            
            # Evaluate model
            try:
                with torch.no_grad():
                    outputs = self.model(input_data)
                    
                    if targets is not None:
                        if hasattr(outputs, 'loss'):
                            objective = -outputs.loss.item()
                        else:
                            loss = F.cross_entropy(outputs.logits, targets)
                            objective = -loss.item()
                    else:
                        # Use perplexity or other metrics
                        if hasattr(outputs, 'logits'):
                            perplexity = torch.exp(F.cross_entropy(outputs.logits.view(-1, outputs.logits.size(-1)), 
                                                                 input_data.view(-1)))
                            objective = -perplexity.item()
                        else:
                            objective = 0.0
                
                objectives.append(objective)
                
            except Exception as e:
                # Handle numerical instabilities
                objectives.append(-1e6)
        
        # Restore original parameters
        for name, original_param in original_params.items():
            param = self.model.get_parameter(name)
            if param is not None:
                param.data = original_param
        
        return torch.tensor(objectives, device=samples.device)


class CEMOptimizer:
    """Cross-Entropy Method optimizer."""
    
    def __init__(self, objective_fn: ObjectiveFunction, config: CEMConfig):
        self.objective_fn = objective_fn
        self.config = config
        self.iteration = 0
        
    def _get_temperature(self) -> float:
        """Get current temperature for annealing."""
        if self.config.temperature_schedule == "exponential":
            decay_rate = np.log(self.config.final_temperature / self.config.initial_temperature) / self.config.max_iterations
            return self.config.initial_temperature * np.exp(decay_rate * self.iteration)
        elif self.config.temperature_schedule == "linear":
            alpha = self.iteration / self.config.max_iterations
            return (1 - alpha) * self.config.initial_temperature + alpha * self.config.final_temperature
        else:  # constant
            return self.config.initial_temperature
    
    def optimize(self, initial_mean: torch.Tensor, initial_std: torch.Tensor, 
                context: Dict[str, Any]) -> Tuple[torch.Tensor, List[float]]:
        """
        Run CEM optimization.
        
        Args:
            initial_mean: Initial distribution mean
            initial_std: Initial distribution std
            context: Context for objective function
            
        Returns:
            Tuple of (best_solution, objective_history)
        """
        distribution = CEMDistribution(initial_mean, initial_std)
        objective_history = []
        best_solution = None
        best_objective = float('-inf')
        
        n_elite = int(self.config.population_size * self.config.elite_fraction)
        
        for iteration in range(self.config.max_iterations):
            self.iteration = iteration
            
            # Sample candidates
            samples = distribution.sample(self.config.population_size)
            
            # Evaluate objectives
            objectives = self.objective_fn(samples, context)
            
            # Select elite samples
            elite_indices = torch.topk(objectives, n_elite).indices
            elite_samples = samples[elite_indices]
            elite_objectives = objectives[elite_indices]
            
            # Update best solution
            current_best_idx = torch.argmax(elite_objectives)
            current_best_objective = elite_objectives[current_best_idx].item()
            
            if current_best_objective > best_objective:
                best_objective = current_best_objective
                best_solution = elite_samples[current_best_idx].clone()
            
            # Update distribution
            distribution.update(elite_samples, self.config.momentum)
            
            # Track progress
            mean_objective = torch.mean(objectives).item()
            objective_history.append(mean_objective)
            
            # Check convergence
            if iteration > 5:
                recent_improvement = objective_history[-1] - objective_history[-6]
                if abs(recent_improvement) < self.config.convergence_threshold:
                    print(f"CEM converged at iteration {iteration}")
                    break
            
            # Optional: Add noise annealing
            temperature = self._get_temperature()
            distribution.std *= temperature
        
        return best_solution, objective_history


class CEMAdapter:
    """
    Cross-Entropy Method Adapter for Transformer-Squared framework.
    
    Provides CEM-based optimization for expert selection and parameter adaptation.
    """
    
    def __init__(self, config: CEMConfig):
        self.config = config
        
    def optimize_expert_selection(self, expert_system, task_embedding: torch.Tensor,
                                hidden_states: Optional[torch.Tensor] = None,
                                target_output: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Optimize expert selection using CEM.
        
        Args:
            expert_system: Expert system instance
            task_embedding: Task embedding
            hidden_states: Optional hidden states
            target_output: Optional target output
            
        Returns:
            Optimized expert weights
        """
        # Initialize objective function
        objective_fn = ExpertSelectionObjective(expert_system, self.config)
        
        # Initialize distribution
        num_experts = expert_system.config.num_experts
        initial_mean = torch.zeros(num_experts, device=task_embedding.device)
        initial_std = torch.ones(num_experts, device=task_embedding.device) * self.config.noise_std
        
        # Setup context
        context = {
            'task_embedding': task_embedding,
            'hidden_states': hidden_states,
            'target_output': target_output
        }
        
        # Run optimization
        optimizer = CEMOptimizer(objective_fn, self.config)
        best_logits, history = optimizer.optimize(initial_mean, initial_std, context)
        
        # Convert to probabilities
        best_weights = F.softmax(best_logits, dim=-1)
        
        return best_weights
    
    def optimize_parameters(self, model, input_data: torch.Tensor, 
                          targets: Optional[torch.Tensor] = None,
                          parameter_names: List[str] = None) -> Dict[str, torch.Tensor]:
        """
        Optimize specific model parameters using CEM.
        
        Args:
            model: Model to optimize
            input_data: Input data for evaluation
            targets: Optional target outputs
            parameter_names: Names of parameters to optimize
            
        Returns:
            Dictionary of optimized parameters
        """
        if parameter_names is None:
            # Default to SVF parameters
            parameter_names = []
            for name, module in model.named_modules():
                if hasattr(module, 'S'):  # SVF singular values
                    parameter_names.append(f"{name}.S")
        
        # Collect parameters
        param_dims = []
        original_params = {}
        
        for name in parameter_names:
            param = model.get_parameter(name)
            if param is not None:
                param_dims.append(param.numel())
                original_params[name] = param.data.clone()
        
        total_dim = sum(param_dims)
        
        if total_dim == 0:
            return {}
        
        # Initialize objective function
        objective_fn = ParameterOptimizationObjective(model, self.config)
        
        # Initialize distribution
        device = input_data.device
        initial_mean = torch.zeros(total_dim, device=device)
        initial_std = torch.ones(total_dim, device=device) * self.config.noise_std
        
        # Flatten original parameters as initial mean
        param_idx = 0
        for name in parameter_names:
            param = model.get_parameter(name)
            if param is not None:
                param_size = param.numel()
                initial_mean[param_idx:param_idx + param_size] = param.data.flatten()
                param_idx += param_size
        
        # Setup context
        context = {
            'input_data': input_data,
            'targets': targets,
            'param_names': parameter_names
        }
        
        # Run optimization
        optimizer = CEMOptimizer(objective_fn, self.config)
        best_params, history = optimizer.optimize(initial_mean, initial_std, context)
        
        # Convert back to parameter dictionary
        optimized_params = {}
        param_idx = 0
        
        for name in parameter_names:
            param = model.get_parameter(name)
            if param is not None:
                param_size = param.numel()
                optimized_params[name] = best_params[param_idx:param_idx + param_size].view(param.shape)
                param_idx += param_size
        
        return optimized_params
    
    def adaptive_expert_training(self, expert_system, training_data: List[Dict[str, Any]],
                                num_iterations: int = 10) -> Dict[str, List[float]]:
        """
        Adaptively train expert system using CEM on multiple tasks.
        
        Args:
            expert_system: Expert system to train
            training_data: List of training examples
            num_iterations: Number of CEM iterations per task
            
        Returns:
            Training metrics
        """
        metrics = {
            'task_objectives': [],
            'expert_diversity': [],
            'convergence_iterations': []
        }
        
        for task_idx, task_data in enumerate(training_data):
            task_embedding = task_data['task_embedding']
            hidden_states = task_data.get('hidden_states')
            target_output = task_data.get('target_output')
            
            # Store original config
            original_max_iter = self.config.max_iterations
            self.config.max_iterations = num_iterations
            
            # Optimize expert selection for this task
            optimized_weights = self.optimize_expert_selection(
                expert_system, task_embedding, hidden_states, target_output
            )
            
            # Apply optimized weights to expert system
            if hasattr(expert_system, 'gating_network'):
                with torch.no_grad():
                    # Update gating network to prefer these weights
                    target_logits = torch.log(optimized_weights + 1e-8)
                    
                    # Simple gradient step toward target
                    current_logits = expert_system.gating_network.gate(task_embedding.unsqueeze(0))
                    loss = F.mse_loss(current_logits, target_logits.unsqueeze(0))
                    
                    # Manual gradient update
                    lr = 0.01
                    for param in expert_system.gating_network.gate.parameters():
                        if param.grad is not None:
                            param.data -= lr * param.grad
            
            # Compute metrics
            diversity = -torch.sum(optimized_weights * torch.log(optimized_weights + 1e-8))
            metrics['expert_diversity'].append(diversity.item())
            
            # Restore config
            self.config.max_iterations = original_max_iter
            
            print(f"Task {task_idx + 1}/{len(training_data)} completed")
        
        return metrics
    
    def get_optimization_summary(self, objective_history: List[float]) -> Dict[str, float]:
        """Get summary statistics for optimization run."""
        if not objective_history:
            return {}
        
        return {
            'initial_objective': objective_history[0],
            'final_objective': objective_history[-1],
            'best_objective': max(objective_history),
            'improvement': objective_history[-1] - objective_history[0],
            'num_iterations': len(objective_history),
            'convergence_rate': np.mean(np.diff(objective_history)[-5:]) if len(objective_history) > 5 else 0.0
        } 