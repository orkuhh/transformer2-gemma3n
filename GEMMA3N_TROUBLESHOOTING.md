# Gemma 3n Model Troubleshooting Guide

This guide addresses common issues when working with Google's Gemma 3n models in the Transformer-Squared framework.

## Quick Summary

**Gemma 3n models are currently in PREVIEW status and have several known limitations:**
- Gated access requiring Google approval (can take weeks)
- Limited to Google AI Edge/LiteRT optimization
- SentencePiece tokenizer compatibility issues
- Preview models may have unstable transformers library support

## Common Issues and Solutions

### 1. Authentication Errors

#### Error: `401 Client Error: Unauthorized` or `GatedRepoError`
```bash
huggingface_hub.utils._errors.GatedRepoError: 401 Client Error. 
Cannot access gated repo for url https://huggingface.co/google/gemma-3n-E4B-it-litert-preview
```

**Solutions:**

1. **Request Access (Primary Solution)**
   ```bash
   # Go to model page and request access
   https://huggingface.co/google/gemma-3n-E4B-it-litert-preview
   ```
   - Click "Request access to this model"
   - Wait for Google approval (can take 2+ weeks)
   - You'll receive an email when approved

2. **Setup Authentication**
   ```bash
   # Method 1: HuggingFace CLI
   huggingface-cli login
   
   # Method 2: Environment Variable
   export HF_TOKEN=your_token_here
   
   # Method 3: Python code
   from huggingface_hub import login
   login(token="your_token_here")
   ```

3. **Verify Access**
   ```python
   from huggingface_hub import HfApi
   api = HfApi()
   try:
       model_info = api.model_info("google/gemma-3n-E4B-it-litert-preview")
       print("✅ Access granted!")
   except Exception as e:
       print(f"❌ Access denied: {e}")
   ```

### 2. SentencePiece Tokenizer Issues

#### Error: `Converting from SentencePiece and Tiktoken failed`
```bash
ValueError: Couldn't instantiate the backend tokenizer from one of: 
(1) a `tokenizers` object, (2) a slow tokenizer instance, 
(3) a `SentencePiece` tokenizer, ...
```

**Solutions:**

1. **Install/Update SentencePiece**
   ```bash
   pip install --upgrade sentencepiece
   pip install --upgrade transformers
   ```

2. **Use Alternative Loading Method**
   ```python
   from transformers import GemmaTokenizer
   
   # Try specific tokenizer class
   tokenizer = GemmaTokenizer.from_pretrained(
       "google/gemma-2b-it",  # Use working fallback
       trust_remote_code=True
   )
   ```

3. **Framework Fallback Handling**
   ```python
   from transformer_squared.utils.model_utils import load_tokenizer_with_fallback
   
   # This automatically handles fallbacks
   tokenizer = load_tokenizer_with_fallback("google/gemma-3n-E4B-it-litert-preview")
   ```

### 3. Model Loading Failures

#### Error: Model architecture incompatibility
```bash
OSError: You are trying to access a gated repo.
RuntimeError: Could not load any compatible tokenizer
```

**Solutions:**

1. **Use Enhanced Model Loader**
   ```python
   from transformer_squared.utils.model_utils import ModelLoader
   
   loader = ModelLoader()
   model, tokenizer, config = loader.load_complete_system(
       model_name="google/gemma-3n-E4B-it-litert-preview",
       torch_dtype="float16",
       device_map="auto",
       trust_remote_code=True
   )
   ```

2. **Manual Fallback**
   ```python
   # If Gemma 3n fails, use these fallbacks:
   fallback_models = [
       "google/gemma-2b-it",
       "google/gemma-1.1-2b-it", 
       "google/gemma-2-2b-it"
   ]
   
   for model_name in fallback_models:
       try:
           model = AutoModelForCausalLM.from_pretrained(model_name)
           break
       except Exception as e:
           continue
   ```

### 4. Memory and Performance Issues

#### Error: CUDA out of memory
```bash
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solutions:**

1. **Use Optimized Settings**
   ```python
   model = AutoModelForCausalLM.from_pretrained(
       model_name,
       torch_dtype=torch.float16,  # Use half precision
       device_map="auto",          # Automatic device placement
       low_cpu_mem_usage=True,     # Reduce CPU memory
       offload_folder="./offload"  # Offload to disk if needed
   )
   ```

2. **Gradient Checkpointing**
   ```python
   model.gradient_checkpointing_enable()
   ```

3. **Batch Size Optimization**
   ```python
   # Use smaller batch sizes
   batch_size = 1  # Start small
   max_length = 512  # Limit sequence length
   ```

### 5. Preview Model Limitations

#### Current Gemma 3n Preview Limitations:
- **Primary Use Case**: Optimized for Google AI Edge/LiteRT (mobile devices)
- **Limited Transformers Support**: Preview models may not work with standard transformers
- **Multimodal Incomplete**: Full multimodal features still in development
- **Community Support**: Limited community implementations

**Recommended Approach:**
```python
# Use this for production workloads:
model_name = "google/gemma-2b-it"  # Stable release

# Use this only for experimentation:
model_name = "google/gemma-3n-E4B-it-litert-preview"  # Preview
```

## Working Examples

### Complete Working Example
```python
#!/usr/bin/env python3
import os
from transformer_squared.utils.model_utils import ModelLoader

def main():
    # 1. Setup authentication
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        print("❌ No HF_TOKEN found. Set with: export HF_TOKEN=your_token")
        return
    
    # 2. Initialize loader
    loader = ModelLoader()
    
    # 3. Load with automatic fallbacks
    try:
        model, tokenizer, config = loader.load_complete_system(
            model_name="google/gemma-3n-E4B-it-litert-preview"
        )
        print("✅ Model loaded successfully!")
        
        # 4. Test generation
        prompt = "The future of AI is"
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(inputs['input_ids'], max_new_tokens=20)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
```

### Environment Setup Script
```bash
#!/bin/bash
# setup_gemma3n.sh

echo "🚀 Setting up Gemma 3n environment..."

# Install dependencies
pip install --upgrade transformers
pip install --upgrade sentencepiece
pip install --upgrade huggingface_hub

# Setup authentication
echo "🔐 Setting up HuggingFace authentication..."
echo "Please enter your HuggingFace token:"
read -s HF_TOKEN
export HF_TOKEN=$HF_TOKEN

# Login to HuggingFace
huggingface-cli login --token $HF_TOKEN

echo "✅ Setup complete!"
echo "📝 Next steps:"
echo "1. Request access at: https://huggingface.co/google/gemma-3n-E4B-it-litert-preview"
echo "2. Wait for Google approval (can take weeks)"
echo "3. Run: python transformer_squared/examples/basic_usage.py"
```

## FAQ

### Q: How long does Gemma 3n access approval take?
**A:** Access approval is manual and can take 2-4 weeks. Google employees review requests individually.

### Q: Can I use Gemma 3n for commercial projects?
**A:** Check the Gemma license terms. Preview models may have additional restrictions.

### Q: What's the difference between Gemma 3n E2B and E4B?
**A:** E2B (2B effective parameters) and E4B (4B effective parameters) are different sizes. E4B is larger and more capable.

### Q: Why use Gemma 3n over regular Gemma?
**A:** Gemma 3n uses novel architecture for mobile deployment and parameter efficiency. For most use cases, stick with regular Gemma models.

### Q: My organization blocks HuggingFace. What can I do?
**A:** You can download models manually and load them locally, but this requires significant setup.

## Alternative Solutions

If Gemma 3n continues to be problematic, consider these alternatives:

1. **Google Gemma 2 (Stable)**
   ```python
   model_name = "google/gemma-2-2b-it"  # Well-tested, stable
   ```

2. **Llama 3.2 (Similar size)**
   ```python
   model_name = "meta-llama/Llama-3.2-3B-Instruct"  # Alternative
   ```

3. **Phi-3 (Microsoft)**
   ```python
   model_name = "microsoft/Phi-3-mini-4k-instruct"  # Good performance
   ```

## Getting Help

1. **Check Issues**: Look at HuggingFace model page discussions
2. **Community Forums**: Ask on HuggingFace forums or Reddit r/MachineLearning
3. **Framework Support**: Create issues in this repository with full error logs

## Status Updates

- **January 2025**: Gemma 3n in preview, limited transformers support
- **Expected**: Full release with better transformers integration coming soon
- **Recommendation**: Use Gemma 2 for production, Gemma 3n for experimentation

---

**Remember**: Gemma 3n is a preview model. For production use, prefer stable releases like `google/gemma-2b-it`. 