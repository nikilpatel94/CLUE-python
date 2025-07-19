from llm_uncertainty_evals.output_sequence_generator import OutputSequenceGenerator
from llm_uncertainty_evals.concepts import ConceptExtractor,ConceptPooler
from llm_uncertainty_evals.uncertainty import UncertaintyCalculator
from llm_uncertainty_evals.datatypes import ModelUncertainty,ContextUsability,Config
from llm_uncertainty_evals.config import os_generatorLLM_config,conceptLLM_config, similarityLM_config

def calculate_contexts_usability(question:str,
                         retrieved_contexts:list[str],
                         generated_output:str|None=None,
                         max_output_sequences:int=1,
                         os_generatorLLM_config:None|Config=os_generatorLLM_config,
                         conceptLLM_config:Config=conceptLLM_config
                         )->ContextUsability:
    # Generate output using Input and Context(Very basic RAG) if it is not supplied
    if generated_output is None or len(generated_output) == 0:
        print("Generating RAG output...")
        generator = OutputSequenceGenerator(llm_config=os_generatorLLM_config,max_output_sequences=max_output_sequences)
        generated_output:str|None = generator.generate(question, retrieved_contexts)[0] # Only stores the first generated output
    # Extract concepts from the generated output
    concept_extractor = ConceptExtractor(llm_config=conceptLLM_config)
    extracted_concepts = concept_extractor.extract_concepts([generated_output]) #extract_concepts() takes list[str] arg
    # Calculate Uncertainty scores with respect to the retrieved context chucks
    uncertainty_calculator = UncertaintyCalculator()
    context_uncertainty_scores:list[float] = uncertainty_calculator.calculate_context_uncertainty(pooled_concepts=extracted_concepts,context_chunks=retrieved_contexts).tolist()
    return ContextUsability(generated_output=generated_output,context_chunks=retrieved_contexts,usability_scores=context_uncertainty_scores)

def calculate_model_uncertainty(question:str,
                                retrieved_contexts:None|list[str]=None,
                                max_output_sequences:int=3,
                                os_generatorLLM_config:Config=os_generatorLLM_config,
                                conceptLLM_config:Config=conceptLLM_config
                                )->ModelUncertainty:
    generator = OutputSequenceGenerator(llm_config=os_generatorLLM_config,max_output_sequences=max_output_sequences)
    output_sequences = generator.generate(question, retrieved_contexts)
    concept_extractor = ConceptExtractor(llm_config=conceptLLM_config)
    extracted_concepts = concept_extractor.extract_concepts(output_sequences)
    concept_pooler = ConceptPooler(llm_config=similarityLM_config)
    pooled_concepts = concept_pooler.get_unique_concepts(concepts_list=extracted_concepts)
    uncertainty_calculator = UncertaintyCalculator()
    uncertainty_scores = uncertainty_calculator.calculate_uncertainty(pooled_concepts=pooled_concepts,output_sequences=output_sequences).tolist()
    return ModelUncertainty(output_sequences=output_sequences,
                            extracted_concepts=extracted_concepts,
                            pooled_concepts=pooled_concepts,
                            uncertainty_scores=uncertainty_scores)