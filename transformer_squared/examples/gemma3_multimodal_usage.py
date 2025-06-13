#!/usr/bin/env python3
"""
Example usage of Gemma 3 4B model with multimodal capabilities.
This script demonstrates how to use the Transformer-Squared framework
with Gemma 3 for both text-only and image+text inputs.
"""

import torch
import logging
from transformer_squared.utils.model_utils import ModelLoader
from PIL import Image
import requests
from io import BytesIO

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_image_from_url(url: str) -> Image.Image:
    """Load an image from URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        return image
    except Exception as e:
        logger.error(f"Failed to load image from {url}: {e}")
        return None

def text_only_example(model, processor):
    """Example of text-only interaction with Gemma 3."""
    print("\n" + "=" * 60)
    print("🔤 Text-Only Example")
    print("=" * 60)
    
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]
        },
        {
            "role": "user", 
            "content": [{"type": "text", "text": "Explain the concept of machine learning in simple terms."}]
        }
    ]
    
    try:
        # Apply chat template and tokenize
        inputs = processor.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=True,
            return_dict=True, 
            return_tensors="pt"
        ).to(model.device)
        
        input_len = inputs["input_ids"].shape[-1]
        
        # Generate response
        with torch.inference_mode():
            generation = model.generate(
                **inputs, 
                max_new_tokens=200, 
                do_sample=True,
                temperature=0.7,
                pad_token_id=processor.tokenizer.eos_token_id
            )
            generation = generation[0][input_len:]
        
        # Decode response
        decoded = processor.decode(generation, skip_special_tokens=True)
        
        print(f"🤖 Gemma 3 Response:")
        print(decoded)
        return True
        
    except Exception as e:
        print(f"❌ Text generation failed: {e}")
        return False

def multimodal_example(model, processor):
    """Example of multimodal (image + text) interaction with Gemma 3."""
    print("\n" + "=" * 60)
    print("🖼️ Multimodal Example (Image + Text)")
    print("=" * 60)
    
    # Use a sample image
    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
    print(f"📷 Loading image from: {image_url}")
    
    image = load_image_from_url(image_url)
    if image is None:
        print("❌ Could not load image, skipping multimodal example")
        return False
    
    print("✅ Image loaded successfully")
    
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant that can analyze images."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "What do you see in this image? Describe it in detail."}
            ]
        }
    ]
    
    try:
        # Apply chat template with image
        inputs = processor.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=True,
            return_dict=True, 
            return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16)
        
        input_len = inputs["input_ids"].shape[-1]
        
        # Generate response
        with torch.inference_mode():
            generation = model.generate(
                **inputs, 
                max_new_tokens=200, 
                do_sample=True,
                temperature=0.7,
                pad_token_id=processor.tokenizer.eos_token_id
            )
            generation = generation[0][input_len:]
        
        # Decode response
        decoded = processor.decode(generation, skip_special_tokens=True)
        
        print(f"🤖 Gemma 3 Vision Response:")
        print(decoded)
        return True
        
    except Exception as e:
        print(f"❌ Multimodal generation failed: {e}")
        return False

def main():
    """Main example function."""
    print("🚀 Gemma 3 4B Multimodal Usage Example")
    print("=" * 60)
    
    # Initialize the model loader
    loader = ModelLoader()
    
    # Load the complete system
    try:
        print("📥 Loading Gemma 3 4B model...")
        model, processor, config = loader.load_complete_system(
            model_name="google/gemma-3-4b-it"
        )
        
        print("✅ Model loaded successfully!")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Processor type: {type(processor).__name__}")
        print(f"   Configuration: {config.model_name}")
        
        # Check if we actually got a processor (multimodal) or just tokenizer
        is_multimodal = hasattr(processor, 'image_processor')
        print(f"   Multimodal capabilities: {'✅ Yes' if is_multimodal else '❌ No (text-only)'}")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Run examples
    print("\n" + "=" * 60)
    print("🧪 Running Examples")
    print("=" * 60)
    
    # Text-only example
    success1 = text_only_example(model, processor)
    
    # Multimodal example (only if we have a processor)
    success2 = False
    if hasattr(processor, 'image_processor'):
        success2 = multimodal_example(model, processor)
    else:
        print("\n⚠️  Skipping multimodal example - processor doesn't support images")
        print("   This usually means a fallback text-only model was loaded")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Example Summary")
    print("=" * 60)
    
    if success1:
        print("✅ Text generation: SUCCESS")
    else:
        print("❌ Text generation: FAILED")
    
    if success2:
        print("✅ Multimodal generation: SUCCESS")
    elif not hasattr(processor, 'image_processor'):
        print("⚠️  Multimodal generation: SKIPPED (not supported)")
    else:
        print("❌ Multimodal generation: FAILED")
    
    if success1 or success2:
        print("\n🎉 Gemma 3 is working! You can now:")
        print("   • Generate text responses")
        if success2:
            print("   • Analyze images with text queries")
            print("   • Use both text and image inputs together")
        print("   • Integrate into your applications")
    else:
        print("\n❌ Examples failed. Check your setup and try again.")

if __name__ == "__main__":
    main() 