from llm_uncertainty_evals.models import Groq_LLM

class OutputSequenceGenerator:
    def __init__(self, llm_config:dict,max_output_sequences:int=3):
        self.llm_config = llm_config    
        self.N = max_output_sequences
        self.O = []
    
    def _make_generation_prompt(self, input:str, contexts:list[str]|None)->str:
        if contexts is None or len(contexts) == 0:
            contexts_str = ""
        else:
            contexts_str = "\n".join(contexts)
        return f"""Input: {input}

External Knowledge: {contexts_str}

Output:"""
    
    def generate(self, input:str, contexts:list[str]|None)->list[str]:
        groq_llm = Groq_LLM(config=self.llm_config)
        for _ in range(self.N):
            output_sequence = groq_llm.generate(prompt=self._make_generation_prompt(input=input,contexts=contexts))
            self.O.append(output_sequence)
        return self.O