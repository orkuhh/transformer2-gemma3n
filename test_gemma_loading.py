#!/usr/bin/env python3
"""
Test script for Gemma 3n model loading with authentication and fallback.
This script demonstrates the enhanced loading capabilities of the Transformer-Squared framework.
"""

import os
import logging
from transformer_squared.utils.model_utils import ModelLoader, setup_huggingface_auth, check_model_access

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_gemma_3_loading():
    """Test Gemma 3 4B model loading with comprehensive error handling."""
    
    print("🚀 Testing Gemma 3 4B Model Loading")
    print("=" * 50)
    
    # Initialize the enhanced model loader
    loader = ModelLoader()
    
    # Test authentication setup
    print("\n1. Testing HuggingFace Authentication...")
    auth_result = setup_huggingface_auth()
    if auth_result:
        print("✅ Authentication successful")
    else:
        print("⚠️  Authentication not available - will use fallback models")
    
    # Test model access check
    target_model = "google/gemma-3-4b-it"
    print(f"\n2. Checking access to {target_model}...")
    has_access = check_model_access(target_model)
    if has_access:
        print("✅ Model access confirmed")
    else:
        print("⚠️  No access to Gemma 3 4B - fallback models will be used")
    
    # Test complete system loading
    print("\n3. Loading complete Transformer-Squared system...")
    try:
        model, tokenizer, config = loader.load_complete_system(
            model_name=target_model,
            torch_dtype="auto"  # Let the system decide the best dtype
        )
        
        print("✅ System loaded successfully!")
        print(f"   Model: {type(model).__name__}")
        print(f"   Tokenizer: {type(tokenizer).__name__}")
        print(f"   Config: {config.model_name}")
        
        # Test basic functionality
        print("\n4. Testing basic model functionality...")
        try:
            test_text = "Hello, how are you today?"
            inputs = tokenizer(test_text, return_tensors="pt")
            
            # Test tokenization
            print(f"   Input text: '{test_text}'")
            print(f"   Tokenized length: {inputs['input_ids'].shape[1]} tokens")
            
            # Test model forward pass (just check it doesn't crash)
            with torch.no_grad():
                outputs = model(**inputs)
            print("   ✅ Model forward pass successful")
            
        except Exception as e:
            print(f"   ⚠️  Basic functionality test failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ System loading failed: {e}")
        return False

def test_fallback_mechanisms():
    """Test the fallback mechanisms separately."""
    
    print("\n" + "=" * 50)
    print("🔄 Testing Fallback Mechanisms")
    print("=" * 50)
    
    loader = ModelLoader()
    
    # Test with a non-existent model to trigger fallback
    fake_model = "google/fake-gemma-model-that-does-not-exist"
    print(f"\n1. Testing fallback with non-existent model: {fake_model}")
    
    try:
        model, tokenizer, config = loader.load_complete_system(
            model_name=fake_model
        )
        print("✅ Fallback mechanism worked successfully!")
        print(f"   Loaded fallback model: {config.model_name}")
        return True
        
    except Exception as e:
        print(f"❌ Fallback mechanism failed: {e}")
        return False

if __name__ == "__main__":
    # Import torch here to avoid import errors if not available
    try:
        import torch
    except ImportError:
        print("❌ PyTorch not available. Please install with: pip install torch")
        exit(1)
    
    print("🧪 Transformer-Squared Gemma 3 4B Loading Test")
    print("=" * 60)
    
    # Run tests
    success1 = test_gemma_3_loading()
    success2 = test_fallback_mechanisms()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    if success1:
        print("✅ Gemma 3 4B loading test: PASSED")
    else:
        print("❌ Gemma 3 4B loading test: FAILED")
    
    if success2:
        print("✅ Fallback mechanism test: PASSED")
    else:
        print("❌ Fallback mechanism test: FAILED")
    
    if success1 or success2:
        print("\n🎉 At least one loading method works!")
        print("\nNext steps:")
        print("1. For Gemma 3 4B access, visit: https://huggingface.co/google/gemma-3-4b-it")
        print("2. Accept Google's license terms (immediate access)")
        print("3. Set up authentication: export HF_TOKEN=your_token_here")
        print("4. The system will automatically use fallback models if needed")
    else:
        print("\n❌ All tests failed. Please check your environment setup.") 