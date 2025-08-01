from transformers import pipeline
from transformers import AutoTokenizer,AutoModelForSequenceClassification,GenerationConfig
import torch
import os
import warnings
from dotenv import load_dotenv
import os
from llm_uncertainty_evals.config import nli_config,torch_config,similarityLM_config
from llm_uncertainty_evals.datatypes import Config
from groq import Groq
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import instructor
from pydantic import BaseModel

warnings.filterwarnings("ignore")
load_dotenv()

class Groq_LLM:
    def __init__(self,config:dict):
        self.config = config

    def s_generate(self,prompt:str,base_model:BaseModel)->BaseModel:
        """
        Generates outputs in Structured format with supplied BaseModel, using the Groq LLM and Instructor API.

        Args:
            prompt (str): User prompt 
            base_model (pydantic.BaseModel): The pydantic BaseModel to be used to get the list of concepts.
        
        Returns:
            pydantic.BaseModel: BaseModel object.
        """
        instruct_client = instructor.from_groq(Groq(api_key=os.environ.get("GROQ_API_KEY")))
        try:
            model = self.config.config["model_name"]
            system_msg = self.config.config["system_msg"]
            temperature = self.config.config["temperature"]
            reply = instruct_client.chat.completions.create(
                        model=model,
                        temperature = temperature,
                        messages=[
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": prompt},
                        ],
                        response_model=base_model,
                    )
        except Exception as e:
            print (f"An error occurred in structured generation: {e}")
            reply = None
        return reply

    def generate(self,prompt:str)->str:
        """
        Generates outputs with normal inference using the Groq LLM.

        Args:
            prompt (str): User prompt 
        
        Returns:
            str: The generated text from the model.
        """
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        try:
            model = self.config.config["model_name"]
            system_msg = self.config.config["system_msg"]
            temperature = self.config.config["temperature"]
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
    def __init__(self,config:dict):
        self.config = config

    def generate(self,prompt:str)->str:
            """
                This function sends a prompt to the OpenAI for generation using gpt-4o-mini at temperature of 0.0 by default .
                
                Args:
                    prompt (str): The input text for the model.
                    model (str): The model to use, default is "gpt-4o-mini".

                Returns:
                    str: The response from the model.
            """      
            try:
                model = self.config["model_name"]
                system_msg = self.config["system_msg"]
                temperature = self.config["temperature"]
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


class NLIModelTorch(torch.nn.Module):
    def __init__(self)->None:
        super().__init__()
        self.settings = nli_config.config
        hf_model_name = self.settings["model_name"]
        local_model_path = self.settings["local_model_path"]
        self.device  = self.settings["device"]
        if not os.path.exists(os.path.join(local_model_path, "config.json")):
            print("Downloading and saving the model...")
            # Create a pipeline to trigger downloading the model
            pipe = pipeline("fill-mask", model=hf_model_name)  # Example pipeline
            pipe.model.save_pretrained(local_model_path)
            pipe.tokenizer.save_pretrained(local_model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(local_model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        classification_config = GenerationConfig(eos_token_id=self.model.config.eos_token_id)
        classification_config.save_pretrained(local_model_path)

    def forward(self,basis_sequence:str,concept:str)->torch.Tensor:
        if "MANUAL_SEED" in self.settings.keys():
            SEED = self.settings["MANUAL_SEED"]
        else:
            SEED = torch_config.config["MANUAL_SEED"]
        torch.manual_seed(SEED)
        x = self.tokenizer.encode(basis_sequence, concept, padding=True,return_tensors='pt',truncation='longest_first').to(self.device)
        logits = self.model(x)[0]
        return logits

    def get_entailment_score(self,output_sequence:str,concept:str)->float:
        logits = self(output_sequence,f"This example is {concept}")
        entail_contradiction_logits = logits[:,[0,2]]
        probs = entail_contradiction_logits.softmax(dim=1)
        prob_label_is_true = probs[:,1][0].to('cpu').item()
        return prob_label_is_true
    
class SimilarityModel:
    def __init__(self,llm_config:Config=similarityLM_config):
        self.config = llm_config
        model_name = self.config.config["model_name"]
        local_model_path = self.config.config["local_model_path"]
        self.device  = self.config.config["device"]
        if not os.path.exists(os.path.join(local_model_path, "config.json")):
            print(f"Downloading and saving Sentence Transformer model {model_name}")
            model = SentenceTransformer(model_name,device=self.device)
            model.save(local_model_path)
        self.model = SentenceTransformer(local_model_path,device=self.device)
        print("Sentence Transformer model is ready for use.")