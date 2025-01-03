# CLUE-python
The python implementation of [CLUE: Concept-Level Uncertainty Estimation for Large Language Models](https://arxiv.org/abs/2409.03021) paper.

## Overview

CLUE can be used to derive an explainable uncertainty in black box LLM generation using NLI. 

Below is the overview of how CLUE works. 

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
- Multilingual evaluations can be unpreditctable to evaluate with the used NLI model
- For detecting RAG hallucinations using this framework, the irrelevant contexts from the retrieved list of contexts have to be removed to get accurate entailment scores and then uncertainty scores

## Immediate Road-map

- [ ] support for openai, ollama, google LLms

