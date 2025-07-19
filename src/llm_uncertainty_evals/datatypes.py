from dataclasses import dataclass
from pydantic import BaseModel

class Concepts(BaseModel):
    concepts:list[str]

@dataclass
class Config:
    config:dict

@dataclass
class ContextUsability:
    generated_output:str
    context_chunks:list[str]
    usability_scores:list[float]

@dataclass
class ModelUncertainty:
    output_sequences:list[str]
    extracted_concepts:list
    pooled_concepts:list[str]
    uncertainty_scores:list[float]
    