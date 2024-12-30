from output_sequence_generator import OutputSequenceGenerator
from concepts import ConceptExtractor,ConceptPooler
from uncertainty import UncertaintyCalculator
from config import os_generatorLLM_config,conceptLLM_config

question = "What is the capital of France?"
contexts = ["Paris is the capital and most populous city of France.",
            "France is the fashion capital of the world especially Paris.",
            "France and India are two large democracies in the world."]

generator = OutputSequenceGenerator(llm_config=os_generatorLLM_config,max_output_sequences=2)
output_sequences = generator.generate(question, contexts)
concept_extractor = ConceptExtractor(llm_config=conceptLLM_config)
extracted_concepts = concept_extractor.extract_concepts(output_sequences)
concept_pooler = ConceptPooler(similarity_threshold=0.98)
pooled_concepts = concept_pooler.get_unique_concepts(concepts_list=extracted_concepts)
uncerainty_calculator = UncertaintyCalculator()
uncertainty_scores = uncerainty_calculator.calculate_uncertainty(pooled_concepts=pooled_concepts,output_sequences=output_sequences)

print("\noutput sequences:",output_sequences)

print("\nextracted concepts:",extracted_concepts)

print("\npooled concepts:",pooled_concepts)

print("\nuncertainty scores",uncertainty_scores)
