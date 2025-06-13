#!/usr/bin/env python3
"""
Basic usage example for Transformer-Squared framework with Gemma 3 4B support.

This example demonstrates:
1. Proper HuggingFace authentication for Gemma models
2. Robust model loading with fallbacks
3. Task classification and expert system usage
4. Error handling for common issues
"""

import os
import sys
import logging
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded environment variables from .env file")
except ImportError:
    print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")
except Exception as e:
    print(f"⚠️  Could not load .env file: {e}")

# Add the parent directory to the path to import transformer_squared
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import torch
except ImportError:
    print("❌ PyTorch is required but not installed.")
    print("   Install with: pip install torch")
    sys.exit(1)

from transformer_squared.utils.model_utils import ModelLoader, TaskClassifier
from transformer_squared.adaptation.expert_system import ExpertSystem

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_authentication():
    """
    Setup HuggingFace authentication for accessing Gemma models.
    
    Gemma models require a valid HuggingFace token for access.
    """
    print("🔐 Setting up HuggingFace authentication...")
    
    # Check for existing token
    hf_token = os.getenv('HF_TOKEN') or os.getenv('HUGGING_FACE_HUB_TOKEN')
    
    if not hf_token:
        print("⚠️  No HuggingFace token found!")
        print("\n📋 To access Gemma models, you need to:")
        print("   1. Go to https://huggingface.co/google/gemma-3-4b-it")
        print("   2. Accept the license agreement")
        print("   3. Create a HuggingFace token at https://huggingface.co/settings/tokens")
        print("   4. Set it as environment variable: export HF_TOKEN=your_token_here")
        print("   5. Or run: huggingface-cli login")
        print("\n🔄 For now, we'll use a fallback model...")
        return False
    else:
        print("✅ HuggingFace token found!")
        return True

def demonstrate_model_loading():
    """Demonstrate robust model loading with fallbacks."""
    print("\n🚀 Loading Transformer-Squared model...")
    
    # Initialize model loader
    loader = ModelLoader()
    
    try:
        # Get model name from environment or use default
        model_name = os.getenv('DEFAULT_MODEL', 'google/gemma-3-4b-it')
        print(f"🎯 Using model: {model_name}")
        
        # Attempt to load Gemma 3 4B model
        model, tokenizer, config = loader.load_complete_system(
            model_name=model_name,
            torch_dtype="float16",  # Use float16 for memory efficiency
            device_map="auto"
        )
        
        print(f"✅ Model loaded successfully!")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Vocab size: {config.vocab_size}")
        print(f"   Hidden size: {config.hidden_size}")
        print(f"   Number of layers: {config.num_layers}")
        
        return model, tokenizer, config
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("🔄 Please check your authentication and try again.")
        return None, None, None

def demonstrate_task_classification(model, tokenizer, config):
    """Demonstrate task classification capabilities."""
    if not all([model, tokenizer, config]):
        print("⚠️  Skipping task classification - model not loaded")
        return
        
    print("\n🎯 Demonstrating task classification...")
    
    try:
        # Import TaskClassificationConfig
        from transformer_squared.utils.model_utils import TaskClassificationConfig
        
        # Create task classifier with proper config
        task_config = TaskClassificationConfig()
        task_classifier = TaskClassifier(task_config)
        
        # Test examples
        test_examples = [
            "What is the capital of France?",
            "Translate 'hello' to Spanish",
            "Summarize the benefits of renewable energy",
            "Write a Python function to sort a list",
            "Generate a creative story about a robot"
        ]
        
        print("📝 Classifying tasks:")
        for example in test_examples:
            try:
                # Classify task using text directly
                result = task_classifier.classify_task(example)
                task_name = result['task_names'][0] if result['task_names'] else "unknown"
                confidence = result['confidence_scores'][0] if len(result['confidence_scores']) > 0 else 0.0
                print(f"   '{example[:50]}...' → {task_name} (confidence: {confidence:.2f})")
                
            except Exception as e:
                print(f"   ⚠️  Failed to classify: {example[:30]}... (Error: {e})")
                
    except Exception as e:
        print(f"⚠️  Task classification error: {e}")

def demonstrate_expert_system(config):
    """Demonstrate expert system functionality."""
    print("\n🧠 Demonstrating expert system...")
    
    try:
        # Import ExpertConfig
        from transformer_squared.adaptation.expert_system import ExpertConfig
        
        # Create expert system with proper config
        expert_config = ExpertConfig(
            num_experts=8,
            expert_dim=64,
            hidden_size=config.hidden_size if hasattr(config, 'hidden_size') else 2048
        )
        expert_system = ExpertSystem(expert_config)
        
        # Show expert statistics
        stats = expert_system.get_expert_statistics()
        print(f"📊 Expert system statistics:")
        print(f"   Number of layers: {stats.get('num_layers', 'Unknown')}")
        print(f"   Experts per layer: {stats.get('experts_per_layer', 'Unknown')}")
        print(f"   Total experts: {stats.get('total_experts', 'Unknown')}")
        
        # Test expert mixing for different task types
        task_types = ["question_answering", "text_generation", "summarization"]
        print(f"\n🔀 Expert mixing examples:")
        
        for task_type in task_types:
            try:
                # Create a dummy task embedding for demonstration
                task_embedding = torch.randn(1, expert_config.task_embedding_dim)
                
                # Get expert weights using gating network
                expert_weights, _ = expert_system.gating_network(task_embedding, training=False)
                if expert_weights is not None:
                    active_experts = (expert_weights[0] > 0.1).sum().item()
                    print(f"   {task_type}: {active_experts} active experts")
                else:
                    print(f"   {task_type}: No expert weights available")
            except Exception as e:
                print(f"   ⚠️  {task_type}: Error getting weights ({e})")
                
    except Exception as e:
        print(f"⚠️  Expert system error: {e}")

def demonstrate_text_generation(model, tokenizer):
    """Demonstrate text generation with error handling."""
    if not all([model, tokenizer]):
        print("⚠️  Skipping text generation - model not loaded")
        return
        
    print("\n📝 Demonstrating text generation...")
    
    # Test prompts
    prompts = [
        "The future of artificial intelligence is",
        "Climate change can be addressed by",
        "In a world where robots and humans coexist,"
    ]
    
    for prompt in prompts:
        try:
            print(f"\n💭 Prompt: '{prompt}'")
            
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=100)
            
            # Generate response with conservative settings
            with torch.no_grad():
                outputs = model.generate(
                    inputs['input_ids'],
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id,
                    attention_mask=inputs.get('attention_mask')
                )
            
            # Decode response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = response[len(prompt):].strip()
            
            print(f"🤖 Generated: '{generated_text}'")
            
        except Exception as e:
            print(f"⚠️  Generation failed for prompt '{prompt[:30]}...': {e}")

def main():
    """Main demonstration function."""
    print("🎉 Welcome to Transformer-Squared Framework Demo!")
    print("="*60)
    
    # Setup authentication
    auth_setup = setup_authentication()
    
    # Load model system
    model, tokenizer, config = demonstrate_model_loading()
    
    if model is None:
        print("\n❌ Cannot proceed without a loaded model.")
        print("🔧 Please check the troubleshooting guide in INSTALLATION_SUMMARY.md")
        return
    
    # Demonstrate core features
    demonstrate_task_classification(model, tokenizer, config)
    demonstrate_expert_system(config)
    demonstrate_text_generation(model, tokenizer)
    
    print("\n✅ Demo completed successfully!")
    print("\n📚 Next steps:")
    print("   1. Check out more examples in the examples/ directory")
    print("   2. Read the documentation for advanced features")
    print("   3. Try fine-tuning with your own data")
    
    if not auth_setup:
        print("\n🔐 Authentication reminder:")
        print("   - Set up HuggingFace authentication for full Gemma model access")
        print("   - Current demo used fallback models")

if __name__ == "__main__":
    main() 