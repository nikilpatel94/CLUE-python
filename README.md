# CLUE-python
The python implementation of [CLUE: Concept-Level Uncertainty Estimation for Large Language Models](https://arxiv.org/abs/2409.03021) paper.

## Overview

CLUE can be used to derive an explainable uncertainty in black box LLM generation using NLI. 

Below is the overview of how CLUE works. 

![Alt text](images/CLUE_diagram.png)

## Models/Platforms

- To generate output sequences and concepts: ![groq](https://groq.com/)
- To generate entailment scores: ![bart-large-mnli](https://huggingface.co/facebook/bart-large-mnli)

## Limitations

- No evaluations from the paper are included in this implementation
- Multilingual evaluations can be difficult to evaluate with the current NLI model
- For detecting RAG hallucinations using this framework, the irrelevant contexts has to be removed to get an accurate entailment scores and then uncertainty scores

## Immediate Road-map

- [ ] support for openai, ollama, google LLms

