import torch
from llm_uncertainty_evals.models import NLIModelTorch
from llm_uncertainty_evals.config import torch_config
import numpy as np

torch.manual_seed(torch_config.config["MANUAL_SEED"])

class UncertaintyCalculator:
    def __init__(self) -> None:
        self.nli_model = NLIModelTorch()
        
    def extract_entailment_scores(self,concepts:list[str],basis_sequences:list)->list:
        """
        Extract entailment scores for each concept in the pooled concepts for each output sequence.

        Args:
            concepts (list[str]): List of pooled concepts.
            output_sequences (list[str]): List of output sequences.

        Returns:
            list: Entailment scores for each concept in the pooled concepts for each output sequence
        """
        try:
            if concepts is []:
                raise ValueError("No Pooled concepts found.")
            else:
                s_i_j = np.empty((len(basis_sequences), len(concepts)))
                for i,o_i in enumerate(basis_sequences):
                    for j,c_j in enumerate(concepts):
                        s_i_j[i][j] = self.nli_model.get_entailment_score(output_sequence = o_i,concept=c_j)
                return s_i_j
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
        s_i_j = self.extract_entailment_scores(concepts=pooled_concepts,basis_sequences=output_sequences)
        U_j = np.mean(-1 * np.log(s_i_j),axis=0)
        return U_j
    
    def calculate_context_uncertainty(self,pooled_concepts:list[str],context_chunks:list[str])->list[float]:
        """
        Calculate uncertainty scores for each context chucks for every concepts provided.

        Args:
            pooled_concepts (list[str]): List of pooled concepts.
            context_chunks (list[str]): List of context chucks.
        
        Returns:
            list: List of uncertainty scores for each output sequence.
        """
        s_i_j = self.extract_entailment_scores(concepts=pooled_concepts,basis_sequences=context_chunks)
        U_j = np.mean(-1 * np.log(s_i_j),axis=1)
        return U_j