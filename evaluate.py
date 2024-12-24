import pandas as pd
from models  import Groq_LLM
from hallucination_detector import detect_hallucination

splits = {'train': 'data/train-00000-of-00001-44cee39b8c9485bf.parquet', 'test': 'data/test-00000-of-00001-ea85434570966ab6.parquet'}
df = pd.read_parquet("hf://datasets/neural-bridge/rag-hallucination-dataset-1000/" + splits["train"])
# print(df.head())
sample = df.iloc[0]

question = sample["question"]
contexts = [sample["context"]]
response = Groq_LLM().generate(system_msg="You are a helpful assistant",
                             prompt=f"""Question:{question}

Knowledge:{contexts}

Answer:""",)

# response = sample["answer"]
print("Question:",question)
print("Contexts:",contexts)
print(response)

clue_calculator = detect_hallucination(output_sequence=response,retrieved_contexts=contexts)
print(f"Entailment Scores:",clue_calculator.entailment_scores_all)
print(f"Uncertainty Scores:",clue_calculator.uncertainty_scores)
print(f"Extracted concepts: {clue_calculator.concepts}")
