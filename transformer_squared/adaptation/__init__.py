"""
Adaptation module for Transformer-Squared framework.
Contains expert system and CEM adaptation components.
"""

from .expert_system import ExpertSystem, ExpertConfig
from .cem_adapter import CEMAdapter

__all__ = ["ExpertSystem", "ExpertConfig", "CEMAdapter"] 