# CLUE-python
The python implementation of [CLUE: Concept-Level Uncertainty Estimation for Large Language Models](https://arxiv.org/abs/2409.03021) paper.

## Overview

CLUE can be used to derive an explainable uncertainty in black box LLM generation using NLI.


In addition to paper's contribution for measuring Concept level uncertainty for LLMs, I introduce a new way of using CLUE and Concept level uncertainty for measuring Context Usability.
Given the Input, Retrieved Contexts and the Generated output, there is an evaluator that measures the uncertainty
in every context chunk with respect to the generated output and provide an uncertainty score against each contexts. The basic idea is the context uncertainty score is likely to be inversely proposal to the context's contribution in the generation and hence shows its usability. 
In simple words: High Uncertainty ≈ Non useful. 

In summary, I have two evaluators:

1. **Vanilla CLUE** - measures Concept Level Uncertainty
2. **Context Usability** - measures context usefulness (i.e. Contribution of each context chunk into its generation)

![Alt text](images/CLUE_diagram.png)

## Models/Platforms

- To generate output sequences and concepts: [groq](https://groq.com/)
- To generate entailment scores: [bart-large-mnli](https://huggingface.co/facebook/bart-large-mnli)

## Environment vars
```var
GROQ_API_KEY=
```

## Demo

Bash/shell
```shell
git clone https://github.com/nikilpatel94/CLUE-python.git
cd CLUE-python
conda create --name CLUE-python python=3.12.0
conda activate CLUE-python
pip install -r requirements.txt
EXPORT GROQ_API_KEY=your_groq_key
python ./src/example.py
```

Windows powershell
```powershell
git clone https://github.com/nikilpatel94/CLUE-python.git
cd CLUE-python
conda create --name CLUE-python python=3.12.0
conda activate CLUE-python
pip install -r requirements.txt
$env:GROQ_API_KEY="your_groq_key"
python .\src\example.py
```

## Limitations

- Evaluations from the paper are not included in this implementation
- Multilingual evaluations can be unpredictable to evaluate with the used NLI model
- Dataset evaluations are pending

## Immediate Road-map

- [x] Fix pooling of concepts
- [ ] Model Agnostic LLM usage - Support for openai, Ollama and other LLMs
- [x] Streamline lib and model imports
- [x] Add Context usability
- [ ] Dataset Validation

