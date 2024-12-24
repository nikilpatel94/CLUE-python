from transformers import pipeline
from transformers import AutoModel, AutoTokenizer,AutoModelForSequenceClassification,GenerationConfig
import torch
import os
import warnings
from dotenv import load_dotenv
import os
from config import NLIModelConfig,GorqLLMConfig,OpenAILLMConfig,TorchSetting
from groq import Groq
from openai import OpenAI
SEED = 42
torch.manual_seed(SEED)
warnings.filterwarnings("ignore")
use_accelerator = True
load_dotenv()

class Groq_LLM:
    def __init__(self):
        self.settings = GorqLLMConfig.config

    # def groq_inference(self,system_msg:str,prompt:str,model:str="llama-3.1-70b-versatile",temp:float=0.0)->str:
    def generate(self,system_msg:str,prompt:str)->str:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        try:
            model = self.settings["model_name"]
            temperature = self.settings["temperature"]
            response = client.chat.completions.create(
                        model=model,
                        temperature = temperature,
                        messages=[
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": prompt},
                        ]
                    )
            reply = response.choices[0].message.content
        except Exception as e:
            print (f"An error occurred: {e}")
            reply = ""
        return reply


class OpenAI_LLM:
    def __init__(self):
        self.settings = OpenAILLMConfig.config

    def generate(self,system_msg:str,prompt:str)->str:
            """
                This function sends a prompt to the OpenAI for generation using gpt-4o-mini at temperature of 0.0 by default .
                
                Args:
                    prompt (str): The input text for the model.
                    model (str): The model to use, default is "gpt-4o-mini".

                Returns:
                    str: The response from the model.
            """      
            try:
                model = self.settings["model_name"]
                temperature = self.settings["temperature"]
                client = OpenAI()    
                response = client.chat.completions.create(
                    model=model,
                    temperature = temperature,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt},
                    ]
                )
                reply = response.choices[0].message.content
                return reply
            
            except Exception as e:
                print (f"An error occurred: {e}")

class NLIModel:
    def __init__(self):
        self.settings = NLIModelConfig.config
        model_name = self.settings["model_name"]
        local_model_path = self.settings["local_model_path"]
        self.device  = self.settings["device"]
        if not os.path.exists(local_model_path):
            print("Downloading and saving the NLI model")
            pipe = pipeline("zero-shot-classification",
                            model="facebook/bart-large-mnli",
                            device=device,
                            multi_label=False, 
                            cache_dir="./model/")
            pipe.model.save_pretrained(local_model_path)
            pipe.tokenizer.save_pretrained(local_model_path)
        else:
            print("Model already exists locally.")
        print("Model and tokenizer are ready for use.")
        self.classifier = pipeline("zero-shot-classification",
                        model=local_model_path,device=self.device,multi_label=False)
    
    def get_entailment_score(self,sequence_to_classify:str,candidate_labels:list[str])->list:
        scores=self.classifier(sequence_to_classify, candidate_labels)
        return scores["scores"]
    

class NLIModelTorch(torch.nn.Module):
    def __init__(self)->None:
        super().__init__()
        self.settings = NLIModelConfig.config
        hf_model_name = self.settings["model_name"]
        local_model_path = self.settings["local_model_path"]
        self.device  = self.settings["device"]
        if not os.path.exists(local_model_path):
            print("Downloading and saving the model...")
            # Create a pipeline to trigger downloading the model
            pipe = pipeline("fill-mask", model=hf_model_name)  # Example pipeline
            pipe.model.save_pretrained(local_model_path)
            pipe.tokenizer.save_pretrained(local_model_path)
        else:
            print("Model directory exists.")
        self.model = AutoModelForSequenceClassification.from_pretrained(hf_model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        classification_config = GenerationConfig(eos_token_id=self.model.config.eos_token_id)
        classification_config.save_pretrained(local_model_path)

    def forward(self,output_sequence:str,concept:str)->torch.Tensor:
        x = self.tokenizer.encode(output_sequence, concept, padding=True,return_tensors='pt',truncation='longest_first').to(self.device)
        logits = self.model(x)[0]
        return logits

    def get_entailment_score(self,output_sequence:str,concept:str)->float:
        logits = self(output_sequence,f"This example is {concept}")
        entail_contradiction_logits = logits[:,[0,2]]
        probs = entail_contradiction_logits.softmax(dim=1)
        prob_label_is_true = probs[:,1][0].to('cpu').item()
        return prob_label_is_true
        
