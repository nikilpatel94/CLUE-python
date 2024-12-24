import torch
import math
from models import NLIModelTorch

SEED =42
torch.manual_seed(SEED)

class UncertaintyCalculatorRAG:
    def __init__(self,concepts:list[str]) -> None:
        self.nli_model = NLIModelTorch()
        self.concepts =concepts
        self.entailment_scores_all = []
        self.uncertainty_scores = []

    def _extract_entailment_scores(self,output_sequences:list[str]):
        for output_sequence in output_sequences:
            entailment_scores_sequence =[]
            for concept in self.concepts:
                score = self.nli_model.get_entailment_score(output_sequence = output_sequence,concept=concept)
                entailment_scores_sequence.append(score)
            self.entailment_scores_all.append(entailment_scores_sequence)

    def calculate_uncertainty(self,output_sequences:list[str]):
        self._extract_entailment_scores(output_sequences=output_sequences)
        U = -1 * sum([math.log(entailment_score[0]) for entailment_score in self.entailment_scores_all]) / len(self.entailment_scores_all)
        self.uncertainty_scores.append(U)