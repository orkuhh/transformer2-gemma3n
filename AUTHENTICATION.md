# 🔐 Authentication Guide for Gemma 3n

This framework uses Google's actual Gemma 3n E4B model from Hugging Face, which requires authentication and license acceptance.

## 🚀 Quick Setup

### 1. Create Hugging Face Account
- Go to [Hugging Face](https://huggingface.co)
- Create an account or log in

### 2. Accept Gemma License
- Visit the [Gemma 3n E4B model page](https://huggingface.co/google/gemma-3n-E4B-it-litert-preview)
- Click "Access repository" and accept Google's license terms
- This is required to access the model files

### 3. Create Access Token
- Go to [Hugging Face Settings > Access Tokens](https://huggingface.co/settings/tokens)
- Create a new token with "Read" permissions
- Copy the token for authentication

### 4. Authenticate in Your Environment

#### Option A: Command Line Login
```bash
huggingface-cli login
# Enter your token when prompted
```

#### Option B: Environment Variable
```bash
export HUGGINGFACE_HUB_TOKEN="your_token_here"
```

#### Option C: Python Code
```python
from huggingface_hub import login
login(token="your_token_here")
```

## 🔍 Verification

Test that authentication worked:

```python
from transformers import AutoTokenizer

# This should work without errors after authentication
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3n-E4B-it-litert-preview")
print("✅ Authentication successful!")
```

## 💡 Usage in Framework

Once authenticated, you can use the framework normally:

```python
from transformer_squared.utils.model_utils import ModelLoader

# This will now work with proper authentication
system = ModelLoader.create_complete_system(
    model_name="google/gemma-3n-E4B-it-litert-preview"
)
```

## 🚨 Troubleshooting

### "Repository not found" or "401 Client Error"
- Make sure you've accepted the license on the model page
- Verify your token has the correct permissions
- Check that you're logged in: `huggingface-cli whoami`

### "Access to model is restricted"
- Visit the model page and click "Access repository"
- Accept Google's usage license terms
- Wait a few minutes for access to be processed

### Still having issues?
- Try logging out and back in: `huggingface-cli logout` then `huggingface-cli login`
- Check [Hugging Face documentation](https://huggingface.co/docs/hub/security-tokens)
- Ensure you're using the latest version of `transformers` and `huggingface_hub`

## 📋 Requirements

Add these to your requirements if not already present:

```
transformers>=4.36.0
huggingface_hub>=0.19.0
```

## 🔒 Security Notes

- Keep your access token secure and never commit it to version control
- Use environment variables or secure credential storage
- Tokens can be revoked and regenerated if compromised 