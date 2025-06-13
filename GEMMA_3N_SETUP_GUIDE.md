# Gemma 3 4B Model Setup Guide

This guide provides comprehensive setup instructions for using the Gemma 3 4B multimodal model with the Transformer-Squared framework.

## 🔍 Model Overview

The Gemma 3 4B model offers several advantages:
1. **Publicly Accessible** - No special approval required, just license agreement
2. **Multimodal Capabilities** - Supports both text and image inputs
3. **Stable Release** - Production-ready, not preview status
4. **Better Performance** - Improved over previous Gemma versions
5. **Large Context Window** - 128K tokens for comprehensive understanding

## ✅ Solutions Implemented

### 1. Multimodal Processor Support

The framework now supports both text and image processing:

```python
from transformer_squared.utils.model_utils import ModelLoader

# Load with automatic processor/tokenizer detection
loader = ModelLoader()
model, processor, config = loader.load_complete_system("google/gemma-3-4b-it")

# Check if multimodal capabilities are available
is_multimodal = hasattr(processor, 'image_processor')
```

### 2. Intelligent Fallback System

The system uses this priority order:
1. `google/gemma-3-4b-it` (primary multimodal model)
2. `google/gemma-2b-it` (fallback text-only)
3. `google/gemma-1.1-2b-it` (alternative fallback)
4. `google/gemma-2-2b-it` (another alternative)
5. `microsoft/DialoGPT-medium` (last resort)

### 3. Enhanced Model Loader

```python
from transformer_squared.utils.model_utils import ModelLoader

# Create enhanced loader with automatic fallback
loader = ModelLoader()

# Load complete system (automatically handles processor/tokenizer and fallback)
model, processor, config = loader.load_complete_system(
    model_name="google/gemma-3-4b-it"
)
```

## 🚀 Quick Start Instructions

### Step 1: Install Dependencies

```bash
# Install or upgrade required packages
pip install --upgrade transformers>=4.36.0
pip install sentencepiece>=0.1.99
pip install accelerate>=0.24.0
pip install huggingface-hub>=0.19.0

# Or install from requirements
pip install -r transformer_squared/requirements.txt
```

### Step 2: Set Up Authentication

Choose one of these methods:

**Method A: Environment Variable**
```bash
export HF_TOKEN=your_huggingface_token_here
```

**Method B: HuggingFace CLI**
```bash
huggingface-cli login
```

**Method C: Python Code**
```python
from huggingface_hub import login
login(token="your_token_here")
```

### Step 3: Accept Gemma License

1. Visit: https://huggingface.co/google/gemma-3-4b-it
2. Click "Access Gemma on Hugging Face" 
3. Log in and accept Google's license terms
4. Access is granted immediately (no waiting period)

### Step 4: Test the System

```bash
# Run the multimodal example script
python transformer_squared/examples/gemma3_multimodal_usage.py
```

## 🔧 Troubleshooting Common Issues

### Issue 1: SentencePiece Import Error

**Error:** `ImportError: No module named 'sentencepiece'`

**Solution:**
```bash
pip install sentencepiece>=0.1.99
# If that fails, try:
pip install --upgrade --force-reinstall sentencepiece
```

### Issue 2: Authentication Failure

**Error:** `401 Unauthorized` or `gated model`

**Solutions:**
1. Verify your HuggingFace token is valid
2. Check if you've been approved for Gemma access
3. The system will automatically use fallback models

### Issue 3: Model Loading Timeout

**Error:** Slow loading or timeouts

**Solutions:**
```python
# Use enhanced loading with specific settings
model, tokenizer, config = loader.load_complete_system(
    model_name="google/gemma-3n-E4B-it-litert-preview",
    torch_dtype=torch.float16,  # Reduces memory usage
    device_map="auto",          # Automatic device placement
    low_cpu_mem_usage=True      # Memory optimization
)
```

### Issue 4: Tokenizer Conversion Errors

**Error:** `Unable to convert tokenizer`

**Solution:** The system automatically uses fallback tokenizers that are compatible.

## 📊 System Status Check

Use this code to check your system status:

```python
from transformer_squared.utils.model_utils import ModelLoader, setup_huggingface_auth

# Check authentication
auth_status = setup_huggingface_auth()
print(f"Authentication: {'✅ Success' if auth_status else '❌ Failed'}")

# Test loading
try:
    loader = ModelLoader()
    model, tokenizer, config = loader.load_complete_system()
    print(f"✅ System loaded: {config.model_name}")
except Exception as e:
    print(f"❌ Loading failed: {e}")
```

## 🎯 What to Expect

### With Gemma 3n Access:
- Direct loading of Gemma 3n model
- Full preview features available
- May have compatibility issues (preview status)

### Without Gemma 3n Access:
- Automatic fallback to Gemma 2b or 1.1
- Full functionality maintained
- No loss of core capabilities

### If All Fails:
- System falls back to DialoGPT-medium
- Basic conversational AI capabilities
- Framework still functional

## 🔄 Automatic Fallback Flow

```
1. Try Gemma 3n (google/gemma-3n-E4B-it-litert-preview)
   ↓ (if access denied)
2. Try Gemma 2b (google/gemma-2b-it)
   ↓ (if failed)
3. Try Gemma 1.1 (google/gemma-1.1-2b-it)
   ↓ (if failed)
4. Try Gemma 2-2b (google/gemma-2-2b-it)
   ↓ (if failed)
5. Use DialoGPT (microsoft/DialoGPT-medium)
```

## 📝 Important Notes

1. **Gemma 3n is in Preview**: Expect potential compatibility issues
2. **Access Takes Time**: Google approval can take weeks
3. **Fallbacks Work**: System remains functional without Gemma 3n
4. **Memory Requirements**: Gemma models need significant GPU memory
5. **Authentication Required**: Even for fallback models in some cases

## 🆘 Support

If you continue to have issues:

1. Run the test script: `python test_gemma_loading.py`
2. Check the logs for specific error messages
3. Verify your authentication setup
4. Try the fallback models directly
5. The enhanced error messages will guide you to solutions

## 🔗 Useful Links

- [Gemma 3n Model Page](https://huggingface.co/google/gemma-3n-E4B-it-litert-preview)
- [HuggingFace Token Setup](https://huggingface.co/settings/tokens)
- [Google AI Edge Documentation](https://ai.google.dev/edge)
- [Transformers Documentation](https://huggingface.co/docs/transformers)

---

The Transformer-Squared framework now handles Gemma 3n loading issues gracefully with comprehensive fallback mechanisms and detailed error reporting. 