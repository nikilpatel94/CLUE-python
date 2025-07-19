from llm_uncertainty_evals.datatypes import Config
import torch
from appdirs import user_cache_dir
import os

APP_NAME = "CLUE-PYTHON"
APP_AUTHOR = "NikilP"

def get_device(verbose:bool=False):
    if torch.backends.mps.is_available():
        device = 'mps' 
    elif torch.cuda.is_available():
        device = 'cuda'
    else: 
        device = "cpu"
    if verbose: print(f"Using {device} for local inference.")
    return device

def get_model_path(model_name: str) -> str:
    cache_dir = user_cache_dir(APP_NAME, APP_AUTHOR)
    model_path = os.path.join(cache_dir, model_name.replace("/", "_"))
    os.makedirs(model_path, exist_ok=True)
    return model_path

nli_config = Config({
    "model_name" : "facebook/bart-large-mnli",
    "local_model_path": get_model_path("facebook/bart-large-mnli"),
    "device":get_device(),
    "MANUAL_SEED": 42
    })

similarityLM_config = Config({
    "model_name" : "all-MiniLM-L6-v2",
    "similarity_threshold": 0.98,
    "local_model_path": get_model_path("all-MiniLM-L6-v2"),
    'device':get_device(),
    })

conceptLLM_config = Config({
        "provider":"groq",
        "model_name": "llama-3.3-70b-versatile",
        # "model_name":"gemma2-9b-it",
        "temperature": 0.0,
        "system_msg": "You are a linguistic expert concept extractor from a given sequence of text. Do not add any extra or new information.",
    }
)

os_generatorLLM_config = Config({
        "provider":"groq",
        "model_name": "llama-3.3-70b-versatile",
        # "model_name":"gemma2-9b-it",
        "temperature": 0.0,
        "system_msg": "You are helpful and harmless and you follow ethical guidelines and promote positive behavior. Use external knowledge to generate the output.",
    }
)

torch_config=Config({
        "MANUAL_SEED":42
    } 
)