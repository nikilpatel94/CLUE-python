from models import NLIModelTorch
from concepts import ConceptExtractorRAG
from uncertainty import UncertaintyCalculatorRAG

nli_model_torch = NLIModelTorch()

def detect_hallucination(output_sequence:list[str],retrieved_contexts:list[str]):
    output_sequence = output_sequence
    concept_extractor = ConceptExtractorRAG()
    extracted_concepts = concept_extractor.extract_concepts(text_sequence=retrieved_contexts)
    clue_calculator =  UncertaintyCalculatorRAG(concepts=extracted_concepts)
    clue_calculator.calculate_uncertainty(output_sequences=output_sequence)
    return clue_calculator

# user_input="When was the first super bowl?",
# response=["The first superbowl was held on Jan 15, 1967","The cricket match was first played in 1877"]
# retrieved_contexts=["The First AFL–NFL World Championship Game was an American football game played on January 15, 1967 at the Los Angeles Memorial Coliseum in Los Angeles."]
                    
# user_input = "What is the capital of France?"
# retrieved_contexts= ["Paris is the capital and most populous city of France.","France is the fashion capital of the world especially Paris."]
# response = ["Paris is the capital of France."]

# clue_calculator = detect_hallucination(output_sequence=response)
# print(clue_calculator.entailment_scores_all)
# print(clue_calculator.uncertainty_scores)
# print(f"Extracted concepts: {clue_calculator.concepts}")
