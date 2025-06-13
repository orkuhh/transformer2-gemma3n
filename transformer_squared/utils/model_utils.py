"""
Model utilities for Transformer-Squared framework.
Contains model loading, task classification, and helper functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForCausalLM, AutoProcessor, GemmaTokenizer, GemmaForCausalLM
from dataclasses import dataclass
import json
import os
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import logging
from huggingface_hub import login, HfApi
import warnings

logger = logging.getLogger(__name__)

@dataclass
class TransformerSquaredConfig:
    """Configuration for Transformer-Squared models."""
    vocab_size: int = 50000
    hidden_size: int = 2048  
    num_layers: int = 24
    num_heads: int = 32
    model_name: str = "gemma-3-4b-it"
    model_type: str = "gemma"
    num_experts: int = 6
    svf_rank: int = 8
    adaptation_layers: list = None
    
    def __post_init__(self):
        if self.adaptation_layers is None:
            self.adaptation_layers = list(range(0, self.num_layers, 2))

@dataclass
class TaskClassificationConfig:
    """Configuration for task classification."""
    embedding_dim: int = 768
    num_task_types: int = 10
    classification_threshold: float = 0.7
    update_frequency: int = 100
    memory_size: int = 1000
    use_clustering: bool = True
    cluster_update_interval: int = 500


class TaskClassifier(nn.Module):
    """
    Task classifier for identifying task types from input text.
    
    Uses both supervised classification and unsupervised clustering
    to identify and adapt to new task types.
    """
    
    def __init__(self, config: TaskClassificationConfig, tokenizer_name: str = "bert-base-uncased"):
        super().__init__()
        self.config = config
        
        # Load encoder model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.encoder = AutoModel.from_pretrained(tokenizer_name)
        
        # Freeze encoder weights (we only extract features)
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Task classification head
        encoder_dim = self.encoder.config.hidden_size
        self.projection = nn.Linear(encoder_dim, config.embedding_dim)
        self.classifier = nn.Sequential(
            nn.Linear(config.embedding_dim, config.embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.embedding_dim // 2, config.num_task_types)
        )
        
        # Task memory for clustering
        self.task_memory = []
        self.task_labels = []
        
        # Clustering
        if config.use_clustering:
            self.clusterer = KMeans(n_clusters=config.num_task_types, random_state=42)
            self.cluster_fitted = False
        
        # Task type mapping
        self.task_type_names = {
            0: "question_answering",
            1: "text_generation", 
            2: "summarization",
            3: "translation",
            4: "classification",
            5: "sentiment_analysis",
            6: "named_entity_recognition",
            7: "text_completion",
            8: "dialogue",
            9: "other"
        }
        
        self.step_count = 0
    
    def extract_text_features(self, text: Union[str, List[str]]) -> torch.Tensor:
        """Extract features from text using the encoder."""
        if isinstance(text, str):
            text = [text]
        
        # Tokenize
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        )
        
        # Move to device
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Extract features
        with torch.no_grad():
            outputs = self.encoder(**inputs)
            # Use CLS token or mean pooling
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                features = outputs.pooler_output
            else:
                # Mean pooling over sequence length
                features = outputs.last_hidden_state.mean(dim=1)
        
        # Project to embedding dimension
        features = self.projection(features)
        
        return features
    
    def classify_task(self, text: Union[str, List[str]], return_embedding: bool = True) -> Dict[str, Any]:
        """
        Classify task type from input text.
        
        Args:
            text: Input text or list of texts
            return_embedding: Whether to return task embedding
            
        Returns:
            Dictionary with classification results
        """
        # Extract features
        features = self.extract_text_features(text)
        
        # Supervised classification
        logits = self.classifier(features)
        probabilities = F.softmax(logits, dim=-1)
        predicted_classes = torch.argmax(probabilities, dim=-1)
        confidence_scores = torch.max(probabilities, dim=-1)[0]
        
        # Unsupervised clustering (if available)
        cluster_predictions = None
        if self.config.use_clustering and self.cluster_fitted:
            features_np = features.detach().cpu().numpy()
            cluster_predictions = self.clusterer.predict(features_np)
        
        results = {
            'predicted_classes': predicted_classes.detach().cpu().numpy(),
            'probabilities': probabilities.detach().cpu().numpy(),
            'confidence_scores': confidence_scores.detach().cpu().numpy(),
            'cluster_predictions': cluster_predictions,
            'task_names': [self.task_type_names.get(int(cls), "unknown") for cls in predicted_classes.detach()]
        }
        
        if return_embedding:
            results['task_embeddings'] = features
        
        # Update memory and clustering
        self._update_memory(features.detach(), predicted_classes.detach())
        
        return results
    
    def _update_memory(self, features: torch.Tensor, labels: torch.Tensor):
        """Update task memory for clustering."""
        features_np = features.detach().cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        for feat, label in zip(features_np, labels_np):
            if len(self.task_memory) >= self.config.memory_size:
                self.task_memory.pop(0)
                self.task_labels.pop(0)
            
            self.task_memory.append(feat)
            self.task_labels.append(label)
        
        self.step_count += 1
        
        # Update clustering periodically
        if (self.config.use_clustering and 
            self.step_count % self.config.cluster_update_interval == 0 and
            len(self.task_memory) > self.config.num_task_types):
            
            self._update_clustering()
    
    def _update_clustering(self):
        """Update clustering model with recent task embeddings."""
        if len(self.task_memory) < self.config.num_task_types:
            return
        
        features_array = np.array(self.task_memory)
        self.clusterer.fit(features_array)
        self.cluster_fitted = True
        
        print(f"Updated clustering with {len(self.task_memory)} samples")
    
    def get_similar_tasks(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Find similar tasks from memory."""
        if not self.task_memory:
            return []
        
        # Get features for query text
        query_features = self.extract_text_features(text)
        query_np = query_features.detach().cpu().numpy()
        
        # Compute similarities
        memory_array = np.array(self.task_memory)
        similarities = cosine_similarity(query_np, memory_array)[0]
        
        # Get top-k most similar
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        similar_tasks = []
        for idx in top_indices:
            similar_tasks.append({
                'index': int(idx),
                'similarity': float(similarities[idx]),
                'task_label': int(self.task_labels[idx]),
                'task_name': self.task_type_names.get(self.task_labels[idx], "unknown")
            })
        
        return similar_tasks
    
    def save_classifier(self, save_path: str):
        """Save classifier state."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        state = {
            'model_state_dict': self.state_dict(),
            'config': self.config.__dict__,
            'task_memory': self.task_memory,
            'task_labels': self.task_labels,
            'step_count': self.step_count,
            'task_type_names': self.task_type_names
        }
        
        if self.config.use_clustering and self.cluster_fitted:
            state['clusterer'] = self.clusterer
        
        with open(save_path, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"Task classifier saved to {save_path}")
    
    def load_classifier(self, save_path: str):
        """Load classifier state."""
        with open(save_path, 'rb') as f:
            state = pickle.load(f)
        
        self.load_state_dict(state['model_state_dict'])
        self.task_memory = state['task_memory']
        self.task_labels = state['task_labels']
        self.step_count = state['step_count']
        self.task_type_names = state['task_type_names']
        
        if 'clusterer' in state:
            self.clusterer = state['clusterer']
            self.cluster_fitted = True
        
        print(f"Task classifier loaded from {save_path}")


def setup_huggingface_auth():
    """Setup HuggingFace authentication for accessing gated models."""
    # Try multiple methods to authenticate
    hf_token = None
    
    # Method 1: Environment variable
    hf_token = os.getenv('HF_TOKEN') or os.getenv('HUGGING_FACE_HUB_TOKEN')
    
    # Method 2: Try to get from HF CLI login
    if not hf_token:
        try:
            from huggingface_hub import HfFolder
            hf_token = HfFolder.get_token()
        except Exception:
            pass
    
    # Method 3: Manual authentication prompt (for interactive use)
    if not hf_token:
        logger.warning("⚠️  No HuggingFace token found. For Gemma 3n access, you need to:")
        logger.warning("   1. Get approved access at https://huggingface.co/google/gemma-3n-E4B-it-litert-preview")
        logger.warning("   2. Set HF_TOKEN environment variable or run: huggingface-cli login")
        return False
    
    try:
        login(token=hf_token, add_to_git_credential=False)
        logger.info("✅ HuggingFace authentication successful")
        return True
    except Exception as e:
        logger.warning(f"⚠️  HuggingFace authentication failed: {e}")
        return False

def check_model_access(model_name: str) -> bool:
    """Check if we have access to a specific model."""
    try:
        api = HfApi()
        # Try to get model info - this will fail if we don't have access
        model_info = api.model_info(model_name)
        return True
    except Exception as e:
        if "gated" in str(e).lower() or "unauthorized" in str(e).lower():
            logger.warning(f"⚠️  Model {model_name} requires special access approval")
            logger.warning("   Visit the model page and request access from Google")
            return False
        else:
            logger.warning(f"⚠️  Cannot access model {model_name}: {e}")
            return False

def load_processor_or_tokenizer_with_fallback(model_name: str, **kwargs):
    """Load processor (for multimodal) or tokenizer with fallback options."""
    try:
        # For Gemma 3 4B (multimodal), use AutoProcessor
        if "gemma-3" in model_name and "4b" in model_name:
            logger.info(f"Loading processor for multimodal model {model_name}...")
            processor = AutoProcessor.from_pretrained(model_name, **kwargs)
            logger.info("✅ Processor loaded successfully")
            return processor
        else:
            # For other models, use tokenizer
            logger.info(f"Loading tokenizer for {model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
            logger.info("✅ Tokenizer loaded successfully")
            return tokenizer
    
    except Exception as e:
        logger.warning(f"⚠️  Failed to load processor/tokenizer for {model_name}: {e}")
        
        # Fallback strategies for Gemma models
        if "gemma" in model_name.lower():
            fallback_models = [
                "google/gemma-3-4b-it",
                "google/gemma-2b-it",
                "google/gemma-1.1-2b-it", 
                "google/gemma-2-2b-it"
            ]
            
            for fallback_model in fallback_models:
                try:
                    logger.info(f"🔄 Trying fallback processor/tokenizer: {fallback_model}")
                    
                    if "gemma-3" in fallback_model and "4b" in fallback_model:
                        processor = AutoProcessor.from_pretrained(fallback_model, **kwargs)
                        logger.info(f"✅ Successfully loaded fallback processor: {fallback_model}")
                        return processor
                    else:
                        tokenizer = AutoTokenizer.from_pretrained(fallback_model, **kwargs)
                        logger.info(f"✅ Successfully loaded fallback tokenizer: {fallback_model}")
                        return tokenizer
                        
                except Exception as fallback_error:
                    logger.warning(f"⚠️  Fallback {fallback_model} also failed: {fallback_error}")
                    continue
        
        # Ultimate fallback
        logger.error(f"❌ All processor/tokenizer loading attempts failed for {model_name}")
        raise RuntimeError(f"Could not load any compatible processor/tokenizer for {model_name}")

def load_tokenizer_with_fallback(model_name: str, **kwargs):
    """Load tokenizer with fallback options for problematic models."""
    try:
        # First attempt: Direct loading
        logger.info(f"Loading tokenizer for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
        logger.info("✅ Tokenizer loaded successfully")
        return tokenizer
    
    except Exception as e:
        logger.warning(f"⚠️  Failed to load tokenizer for {model_name}: {e}")
        
        # Fallback strategies for Gemma models
        if "gemma" in model_name.lower():
            fallback_models = [
                "google/gemma-3-4b-it",
                "google/gemma-2b-it",
                "google/gemma-1.1-2b-it", 
                "google/gemma-2-2b-it"
            ]
            
            for fallback_model in fallback_models:
                try:
                    logger.info(f"🔄 Trying fallback tokenizer: {fallback_model}")
                    tokenizer = AutoTokenizer.from_pretrained(fallback_model, **kwargs)
                    logger.info(f"✅ Successfully loaded fallback tokenizer: {fallback_model}")
                    return tokenizer
                except Exception as fallback_error:
                    logger.warning(f"⚠️  Fallback {fallback_model} also failed: {fallback_error}")
                    continue
        
        # Ultimate fallback
        logger.error(f"❌ All tokenizer loading attempts failed for {model_name}")
        raise RuntimeError(f"Could not load any compatible tokenizer for {model_name}")

def load_model_with_enhanced_fallback(model_name: str, **kwargs):
    """Load model with enhanced fallback and authentication handling."""
    
    # Setup authentication first
    auth_success = setup_huggingface_auth()
    
    # Check model access
    has_access = check_model_access(model_name)
    
    if not has_access and not auth_success:
        logger.warning(f"⚠️  Cannot access {model_name} - using fallback model")
    
    # Attempt to load the requested model
    if has_access or auth_success:
        try:
            logger.info(f"Loading model: {model_name}")
            
            # Special handling for Gemma 3 models (multimodal)
            if "gemma-3" in model_name and "4b" in model_name:
                logger.info("✨ Loading Gemma 3 4B model with multimodal capabilities")
                logger.info("   This model supports both text and image inputs")
                
                # Optimized settings for Gemma 3 4B
                kwargs.update({
                    'torch_dtype': torch.bfloat16,  # Recommended for Gemma 3
                    'device_map': 'auto',
                    'low_cpu_mem_usage': True,
                })
            elif "gemma-3n" in model_name:
                logger.warning("⚠️  Gemma 3n models are in preview and may have compatibility issues")
                logger.warning("   These models are optimized for Google AI Edge/LiteRT")
                
                # Try with specific transformers settings
                kwargs.update({
                    'torch_dtype': torch.float16,
                    'device_map': 'auto',
                    'trust_remote_code': True,
                    'low_cpu_mem_usage': True,
                })
            
            model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
            logger.info(f"✅ Successfully loaded model: {model_name}")
            return model
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to load {model_name}: {e}")
            if "gated" in str(e).lower():
                logger.warning("   This is a gated model - access approval required")
            elif "sentencepiece" in str(e).lower():
                logger.warning("   SentencePiece tokenizer compatibility issue detected")
    
    # Fallback model selection
    fallback_models = [
        "google/gemma-3-4b-it",       # Primary Gemma 3 model
        "google/gemma-2b-it",         # Gemma 2 fallback
        "google/gemma-1.1-2b-it",     # Gemma 1.1 fallback
        "google/gemma-2-2b-it",       # Alternative Gemma 2
        "microsoft/DialoGPT-medium",  # Last resort
    ]
    
    for fallback_model in fallback_models:
        try:
            logger.info(f"🔄 Trying fallback model: {fallback_model}")
            
            # Check access to fallback model
            if not check_model_access(fallback_model):
                continue
                
            # Adjust kwargs for fallback models
            fallback_kwargs = kwargs.copy()
            if 'trust_remote_code' in fallback_kwargs:
                del fallback_kwargs['trust_remote_code']  # Most models don't need this
            
            model = AutoModelForCausalLM.from_pretrained(fallback_model, **fallback_kwargs)
            logger.info(f"✅ Successfully loaded fallback model: {fallback_model}")
            return model
            
        except Exception as fallback_error:
            logger.warning(f"⚠️  Fallback {fallback_model} failed: {fallback_error}")
            continue
    
    # If all fallbacks fail
    logger.error("❌ All model loading attempts failed")
    raise RuntimeError(f"Could not load {model_name} or any fallback models")

class ModelLoader:
    """Enhanced model loader with better error handling and authentication."""
    
    def __init__(self):
        self.setup_environment()
    
    def setup_environment(self):
        """Setup the environment for model loading."""
        # Install required packages if missing
        try:
            import sentencepiece
        except ImportError:
            logger.warning("⚠️  sentencepiece not found. Install with: pip install sentencepiece")
        
        # Setup authentication
        setup_huggingface_auth()
    
    def load_complete_system(self, model_name: str = None, **kwargs):
        """Load complete Transformer-Squared system with enhanced error handling."""
        if model_name is None:
            model_name = "google/gemma-3-4b-it"
        
        logger.info(f"🚀 Loading Transformer-Squared system with {model_name}")
        
        # Load processor (for multimodal) or tokenizer
        try:
            tokenizer = load_processor_or_tokenizer_with_fallback(model_name, **kwargs)
        except Exception as e:
            logger.error(f"❌ Failed to load processor/tokenizer: {e}")
            raise
        
        # Load model
        try:
            model = load_model_with_enhanced_fallback(model_name, **kwargs)
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
        
        # Update configuration based on loaded model
        config = self._create_compatible_config(model, tokenizer, model_name)
        
        logger.info("✅ Complete system loaded successfully")
        return model, tokenizer, config
    
    def _create_compatible_config(self, model, tokenizer_or_processor, model_name):
        """Create a compatible configuration for the loaded model."""
        try:
            # Get model configuration
            model_config = model.config
            
            # Create Transformer-Squared config
            config = TransformerSquaredConfig()
            
            # Map model config attributes
            if hasattr(model_config, 'vocab_size'):
                config.vocab_size = model_config.vocab_size
            elif hasattr(tokenizer_or_processor, 'vocab_size'):
                config.vocab_size = tokenizer_or_processor.vocab_size
            elif hasattr(tokenizer_or_processor, 'tokenizer') and hasattr(tokenizer_or_processor.tokenizer, 'vocab_size'):
                # For processor with embedded tokenizer
                config.vocab_size = tokenizer_or_processor.tokenizer.vocab_size
            else:
                config.vocab_size = len(tokenizer_or_processor) if tokenizer_or_processor else 50000
            
            if hasattr(model_config, 'hidden_size'):
                config.hidden_size = model_config.hidden_size
            elif hasattr(model_config, 'd_model'):
                config.hidden_size = model_config.d_model
            else:
                config.hidden_size = 2048  # reasonable default
            
            if hasattr(model_config, 'num_hidden_layers'):
                config.num_layers = model_config.num_hidden_layers
            elif hasattr(model_config, 'num_layers'):
                config.num_layers = model_config.num_layers
            else:
                config.num_layers = 24  # reasonable default
            
            if hasattr(model_config, 'num_attention_heads'):
                config.num_heads = model_config.num_attention_heads
            elif hasattr(model_config, 'num_heads'):
                config.num_heads = model_config.num_heads
            else:
                config.num_heads = config.hidden_size // 64  # reasonable default
            
            # Set model-specific attributes
            config.model_name = model_name
            config.model_type = getattr(model_config, 'model_type', 'unknown')
            
            logger.info(f"📋 Configuration created: vocab_size={config.vocab_size}, "
                       f"hidden_size={config.hidden_size}, num_layers={config.num_layers}")
            
            return config
            
        except Exception as e:
            logger.warning(f"⚠️  Error creating config: {e}. Using defaults.")
            return TransformerSquaredConfig()
    
    def load_svf_trainer(self, model, config):
        """Load SVF trainer if applicable."""
        try:
            from ..training.svf_trainer import SVFTrainer
            
            # Check if model has parameters that can be adapted with SVF
            svf_parameters = [p for p in model.parameters() if 'svf' in p.name if hasattr(p, 'name')]
            
            if not svf_parameters:
                # Check for any trainable parameters
                trainable_params = [p for p in model.parameters() if p.requires_grad]
                if not trainable_params:
                    logger.warning("⚠️  No trainable parameters found for SVF training")
                    return None
                
                # Use a subset of parameters for SVF if no specific SVF parameters exist
                logger.info("🔧 No SVF-specific parameters found, using subset of model parameters")
                svf_parameters = trainable_params[:min(len(trainable_params), 10)]  # Limit to prevent memory issues
            
            if svf_parameters:
                trainer = SVFTrainer(model, config)
                logger.info(f"✅ SVF trainer created with {len(svf_parameters)} parameters")
                return trainer
            else:
                logger.warning("⚠️  No suitable parameters found for SVF trainer")
                return None
                
        except ImportError as e:
            logger.warning(f"⚠️  SVF trainer not available: {e}")
            return None
        except Exception as e:
            logger.warning(f"⚠️  Failed to create SVF trainer: {e}")
            return None


def compute_model_stats(model) -> Dict[str, Any]:
    """Compute statistics about a model."""
    stats = {
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'model_size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    }
    
    # SVF-specific stats
    svf_params = 0
    svf_layers = 0
    for name, module in model.named_modules():
        if hasattr(module, 'U') and hasattr(module, 'S') and hasattr(module, 'V'):
            svf_layers += 1
            svf_params += module.U.numel() + module.S.numel() + module.V.numel()
    
    stats['svf_parameters'] = svf_params
    stats['svf_layers'] = svf_layers
    stats['svf_efficiency'] = svf_params / stats['total_parameters'] if stats['total_parameters'] > 0 else 0
    
    return stats


def format_model_summary(model) -> str:
    """Create a formatted summary of a model."""
    stats = compute_model_stats(model)
    
    summary = f"""
Transformer-Squared Model Summary
================================
Total Parameters: {stats['total_parameters']:,}
Trainable Parameters: {stats['trainable_parameters']:,}
Model Size: {stats['model_size_mb']:.2f} MB

SVF Adaptation:
- SVF Parameters: {stats['svf_parameters']:,}
- SVF Layers: {stats['svf_layers']}
- SVF Efficiency: {stats['svf_efficiency']:.2%}

Model Type: {type(model).__name__}
"""
    
    return summary 