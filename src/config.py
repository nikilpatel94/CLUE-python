from datatypes import Config
import torch

def get_device(verbose:bool=False):
    if torch.backends.mps.is_available():
        device = 'mps' 
    elif torch.cuda.is_available():
        device = 'cuda'
    else: 
        device = "cpu"
    if verbose: print(f"Using {device} for local inference.")
    return device

nli_config = Config({
    "model_name" : "facebook/bart-large-mnli",
    "local_model_path": "./model/facebook/bart-large-mnli",
    "device":get_device(),
    "MANUAL_SEED": 42
    })

similarityLM_config = Config({
    "model_name" : "all-MiniLM-L6-v2",
    "similarity_threshold": 0.98,
    "local_model_path": "./model/st/all-MiniLM-L6-v2",
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