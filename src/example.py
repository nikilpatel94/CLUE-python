from output_sequence_generator import OutputSequenceGenerator
from concepts import ConceptExtractor,ConceptPooler
from uncertainty import UncertaintyCalculator
from config import os_generatorLLM_config,conceptLLM_config

def run_demo(question:str,contexts:list[str])->None:
    generator = OutputSequenceGenerator(llm_config=os_generatorLLM_config,max_output_sequences=3)
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


print("----Simple Question---:")
question1 = "What is the capital of France?"
contexts1 = ["Paris is the capital and most populous city of France.",
            "France is the fashion capital of the world especially Paris.",
            "France and India are two large democracies in the world."]
run_demo(question1,contexts1)

print("----Tricky Question---:")
question2 = "Who won Border-Gavaskar trophy in 2023??"
contexts2 = [
"""
The Border–Gavaskar Trophy is one of the premier bilateral trophies in Test cricket. Both teams have a reputation of being difficult to beat at home. 
This is borne out by India winning 8 out of 9 series held in India, and Australia winning 4 out of 7 series held in Australia, as of the conclusion of the 2022–23 series. 
The away wins achieved by Australia (2004–05) and India (2018–19 and 2020–21) have earned places in cricket folklore. Both teams have achieved similar number of Test and series wins, and the trophy has changed hands frequently. 
The competitiveness of the series is also reflected in that in both 2000–01 and 2007–08, it was India who ended Australian streaks of 16 consecutive Test wins. 
The 2000–01 series was labelled as the "final frontier" for Australia by their captain Steve Waugh due to the difficulty of winning in India, and was closely fought on both sides. 
"""]
run_demo(question2,contexts2)

