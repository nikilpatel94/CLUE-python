def get_device():
    import torch
    if torch.backends.mps.is_available():
        device = 'mps' 
        print("Using mps for NLI inference.")
    elif torch.cuda.is_available():
        device = 'cuda'
        print("Using cuda for NLI inference.")
    else: 
        device = "cpu"
        print("Neither mps nor cuda found. Continuing with CPU for NLI inference.")
    return device

class NLIModelConfig:
    config = {
    "model_name" : "facebook/bart-large-mnli",
    "local_model_path": "./model/facebook/bart-large-mnli",
    "device": get_device()
    }

class GorqLLMConfig:
    config = {
        "model_name": "llama-3.1-70b-versatile",
        "temperature": 0.0
    }

class OpenAILLMConfig:
    config = {
        "model_name": "gpt-4o-mini",
        "temperature": 0.0
    }

class TorchSetting:
    config={
        "MANUAL_SEED":42
    } 
