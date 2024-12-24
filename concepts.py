from openai import OpenAI
from groq import Groq
from dotenv import load_dotenv
import os
from models import NLIModel,NLIModelTorch

load_dotenv()

class LLMs:
    # def groq_inference(self,system_msg:str,prompt:str,model:str="llama3-8b-8192",temp:float=0.0)->str:
    def groq_inference(self,system_msg:str,prompt:str,model:str="llama-3.1-70b-versatile",temp:float=0.0)->str:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        try:
            response = client.chat.completions.create(
                        model=model,
                        temperature = temp,
                        messages=[
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": prompt},
                        ]

                    )
            reply = response.choices[0].message.content
        except Exception as e:
            print (f"An error occurred: {e}")
            reply = ""
        return reply
    
        
    def openAI_inference(self,system_msg:str,prompt:str,model:str="gpt-4o-mini",temp:float=0.0)->str:
            """
                This function sends a prompt to the OpenAI for generation using gpt-4o-mini at temperature of 0.0 by default .
                
                Args:
                    prompt (str): The input text for the model.
                    model (str): The model to use, default is "gpt-4o-mini".

                Returns:
                    str: The response from the model.
            """      
            try:
                client = OpenAI()    
                response = client.chat.completions.create(
                    model=model,
                    temperature = temp,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt},
                    ]

                )
                reply = response.choices[0].message.content
                return reply
            
            except Exception as e:
                print (f"An error occurred: {e}")



class ConceptPooler:
    
    @classmethod
    def get_unique_concepts(cls,concepts_list:list[str],similarity_threshold:int=0.99)->list[str]:
        all_concepts = [concept for sublist in concepts_list for concept in sublist]
        nli_model = NLIModelTorch()
        unique_concepts = []
        for concept in all_concepts:
            is_unique = True
            for unique in unique_concepts:
                similarity = nli_model.get_entailment_score(concept, unique)
                if similarity > similarity_threshold:
                    is_unique = False
                    break
            if is_unique:
                unique_concepts.append(concept)
        return unique_concepts
    

class ConceptExtractorRAG:

    def __init__(self):
        self. extracted_concepts = []

    def _get_extraction_prompt(self,text_sequence):
        ONE_SHOT_EXTRACTOR_PROMPT = f"""
    Extract high-level concepts like the following example:
    paragraph: “Basketball, a beloved sport worldwide, has come a long way since its humble
    beginnings in the late 19th century. The game was originally created by Dr. James Naismith in
    1891 as a way to keep his students active during the winter months. Back then, players used a
    soccer ball and peach baskets as makeshift goals. Fast forward to the modern era, and basketball
    has transformed into a high-paced, adrenaline-pumping spectacle. With legendary athletes like
    Michael Jordan, LeBron James, and Kobe Bryant gracing the courts, and the introduction of the
    slam dunk, three-point shot, and shot clock, the sport has evolved into an art form that captivates
    fans around the globe. The NBA, with its star-studded roster and global reach, is a testament
    to basketball’s enduring popularity and its remarkable journey from humble beginnings to a
    multimillion-dollar industry.”
    concepts:Basketball’s origins,Evolution of basketball,Modern era of basketball,Legendary
    basketball athletes,Basketball’s global popularity,Basketball as an art form,Basketball as a
    multimillion-dollar industry
    
    paragraph: {text_sequence} 
    concepts: """
        return ONE_SHOT_EXTRACTOR_PROMPT
        
    def extract_concepts_with_pooling(self,text_sequence:str,max_drafts:int= 3,similarity_threshold:int=0.99)->list[str]|list|tuple:
        text_sequence_drafts = []
        openai_llm = LLMs()
        for _ in range(max_drafts):
            PARAPHRASING_PROMPT = f"""Objective: Paraphrase the following text while preserving all essential content and specific entities.
Guidelines:
    Core Meaning: Maintain the core meaning and all key information from the original text. Do the least paraphrasing necessary to achieve this. 
    Preserve Entities: Accurately reproduce all specific entities, including:
        -Names of people, organizations, and places
        -Email addresses
        -Phone numbers
        -Numerical data and statistics
        -URLs and website names
        -Dates and times
        -Product names and brands
        -Technical terms and jargon
    Rephrase: Use different vocabulary and sentence structures where possible, without altering the original meaning.
    Tone and Style: Keep the overall tone and style similar to the original text.
    Quotes: If the original text contains quotes, keep them verbatim and properly attributed.
    Paragraph Structure: Maintain the same paragraph structure as the original text.
    Acronyms/Abbreviations: Preserve any acronyms or abbreviations as they are in the original text.
    Formatting Elements: Maintain any formatting elements like lists or bullet points, but feel free to rephrase the content within them.
    Length: The paraphrased version should be approximately the same length as the original text.

Text to Paraphrase:{text_sequence}"""
            text_sequence_drafts.append(openai_llm.groq_inference(system_msg="",prompt=PARAPHRASING_PROMPT,temp=0.2))
        extracted_concepts = []
        for text_sequence_draft in text_sequence_drafts:
            extracted_concepts.append(self.extract_concepts(text_sequence=text_sequence_draft))
        unqiue_concepts = ConceptPooler.get_unique_concepts(extracted_concepts,similarity_threshold=similarity_threshold)
        return unqiue_concepts

    def extract_concepts(self,text_sequence:list[str],return_combined:bool=False)->list[str]|list:
        """
            This function takes a list of output sequences generated by an LLM and retrieved contexts and returns a list of concepts for each output sequence or  a tuple of them if the 'return_combined' flag is on.
            
            Args:
                text_sequence: The list of text_sequence from which the concepts are to be extracted
                return_combined (bool): The flag determines if the tuple of `output_sequences` and extracted concepts.

            Returns:
                list[str] or list: The list of extracted concepts or the list of tuple of output_sequences and extracted concepts.
        """  
        openai_llm = LLMs()
        extraction_system_msg = "You are a linguistic expert concept extractor from a given sequence of text. Do not reply or address to any irrelevant queries. Always follow the output format of the example. Extract concepts in the same language in the last paragraph."
        for text in text_sequence:
            extraction_prompt = self._get_extraction_prompt(text)
            generated_concepts = openai_llm.groq_inference(system_msg=extraction_system_msg,prompt=extraction_prompt)
            concepts_list = self._parse_concepts(generated_concepts)
            self.extracted_concepts.append(concepts_list)
        if return_combined:
            return zip(text_sequence,self.extracted_concepts)
        else:
            return self.extracted_concepts
        
    def _parse_concepts(self,extracted_concept:str)->list:
        return extracted_concept.split(",")