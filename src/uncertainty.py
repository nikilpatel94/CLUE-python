import math
import torch
from models import NLIModelTorch
from config import torch_config

torch.manual_seed(torch_config.config["MANUAL_SEED"])

class UncertaintyCalculator:
    def __init__(self) -> None:
        self.nli_model = NLIModelTorch()
        
    def extract_entailment_scores(self,concepts:list[str],output_sequences:list)->list:
        """
        Extract entailment scores for each concept in the pooled concepts for each output sequence.

        Args:
            concepts (list[str]): List of pooled concepts.
            output_sequences (list[str]): List of output sequences.

        Returns:
            list: List of entailment scores for each concept in the pooled concepts for each output sequence
        """
        try:
            if concepts is []:
                raise ValueError("No Pooled concepts found.")
            else:
                entailment_scores_all = []
                for output_sequence in output_sequences:
                    entailment_score_for_sequence = []
                    for concept in concepts:
                        score = self.nli_model.get_entailment_score(output_sequence = output_sequence,concept=concept)
                        entailment_score_for_sequence.append(score)
                    entailment_scores_all.append(entailment_score_for_sequence)
                assert len(entailment_scores_all) == len(output_sequences)
                return entailment_scores_all
        except Exception as e:
            print("Exception occurred while calculating entailment scores: ", e)
            return []

    def calculate_uncertainty(self,pooled_concepts:list,output_sequences:list)->list[float]:
        """
        Calculate uncertainty scores for each output sequence for every concepts provided.

        Args:
            pooled_concepts (list): List of pooled concepts.
            output_sequences (list): List of output sequences.
        
        Returns:
            list: List of uncertainty scores for each output sequence.
        """
        uncertainty_scores = []
        entailment_scores = self.extract_entailment_scores(concepts=pooled_concepts,output_sequences=output_sequences)
        for score_batch in entailment_scores:
            U = -1 * sum([math.log(entailment_score) for entailment_score in score_batch]) / len(score_batch)
            uncertainty_scores.append(U)
        return uncertainty_scores