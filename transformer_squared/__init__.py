"""
Transformer-Squared: Self-adaptive LLMs with Gemma 3n Support

This package implements the Transformer² framework for self-adaptive large language models,
featuring Singular Value Fine-tuning (SVF) and dynamic expert mixing capabilities.
Now with support for Google's Gemma 3n model.
"""

from .models.gemma3n_model import Gemma3nTransformerSquared
from .svf.singular_value_finetuning import SVFTrainer, SVFConfig
from .adaptation.expert_system import ExpertSystem, ExpertConfig
from .adaptation.cem_adapter import CEMAdapter
from .utils.model_utils import ModelLoader, TaskClassifier

__version__ = "1.0.0"
__all__ = [
    "Gemma3nTransformerSquared",
    "SVFTrainer", 
    "SVFConfig",
    "ExpertSystem",
    "ExpertConfig", 
    "CEMAdapter",
    "ModelLoader",
    "TaskClassifier"
] 