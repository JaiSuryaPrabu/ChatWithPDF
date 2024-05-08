# This file contains all the functionalities from the pdf extraction to the embeddings
import os
import re

from tqdm import tqdm
from spacy.lang.en import English
import fitz
import pandas as pd

import torch
from sentence_transformers import SentenceTransformer

class Embeddings:

    def __init__(self,pdf_file_path : str):
        self.pdf_file_path = pdf_file_path
        self.embedding_model_name = "all-mpnet-base-v2"
        self.device = self.get_device()
            
    def get_device(self) -> str:
        """ Returns the device """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return device

    def text_formatter(self,text : str) -> str:
        """ Convert the text that contains the /n with the space"""
        formatted_text = text.replace('\n',' ').strip()
    
        return formatted_text

    def count_and_split_sentence(self,text : str) -> (int,list[str]):
        """To count and split the sentences from the given text """
        nlp = English()
        nlp.add_pipe("sentencizer")

        list_of_sentences = list(nlp(text).sents)
        list_of_sentences = [str(sentence) for sentence in list_of_sentences]

        return len(list_of_sentences),list_of_sentences

    def open_pdf(self):
        """convert the pdf into dict dtype"""
        doc = fitz.open(self.pdf_file_path)
        data = []

        print("[INFO] Converting the pdf into dict dtype")
        for page_number,page in tqdm(enumerate(doc)):
            text = page.get_text()
            text = self.text_formatter(text = text)

            sentence_count,sentences = self.count_and_split_sentence(text)

            data.append(
                {
                    "page_number" : page_number,
                    "char_count" : len(text),
                    "word_count" : len(text.split(" ")),
                    "sentence_count" : sentence_count,
                    "token_count" : len(text) / 4,
                    "sentence" : sentences,
                    "text" : text
                }
            )

        return data

    def split_the_array(self,array_list : list,
                    chunk_length : int) -> list[list[str]]:
        """Split the array of sentences into groups of chunks"""
        return [array_list[i:i+chunk_length] for i in range(0,len(array_list),chunk_length)]

    def convert_to_chunk(self,chunk_size : int = 10) -> list[dict]:
        """ Convert the sentences into chunks """
        pages_and_texts = self.open_pdf()
        pages_and_chunks = []

        # splitting the chunks 
        print("[INFO] Splitting the sentences ")
        for item in tqdm(pages_and_texts):
            item["sentence_chunks"] = self.split_the_array(item["sentence"],chunk_size)
            item["chunk_count"] = len(item["sentence_chunks"])
    
        # splitting the chunks
        print("[INFO] Splitting into chunks ")
        for item in tqdm(pages_and_texts):
            for chunks in item["sentence_chunks"]:
                d = {}

                joined_sentence = "".join(chunks).replace("  "," ").strip()
                joined_sentence = re.sub(r'\.([A-Z])', r'. \1',joined_sentence) # .A -> . A it is used to provide a space after a sentence ends

                if len(joined_sentence) / 4 > 30:
                    d["page_number"] = item["page_number"]
                    d["sentence_chunk"] = joined_sentence
                    # stats
                    d["char_count"] = len(joined_sentence)
                    d["word_count"] = len(list(joined_sentence.split(" ")))
                    d["token_count"] = len(joined_sentence) / 4 # 4 tokens ~ 1 word
        
                    pages_and_chunks.append(d)
    
        return pages_and_chunks

    def convert_to_embedds(self,chunk_size = 10) -> list[dict] :
    
        data = self.convert_to_chunk(chunk_size)
        
        embedding_model = SentenceTransformer(model_name_or_path = self.embedding_model_name,device = self.device)
        print("[INFO] Converting into embeddings ")
        for item in tqdm(data):
            item["embeddings"] = embedding_model.encode(item["sentence_chunk"], convert_to_tensor = True)
    
        return data

    def save_the_embeddings(self,filename : str = "embeddings.csv",data : list[dict] = None):
        embedd_file = filename
        if data is None:
            data = self.convert_to_embedds()
        dataframe = pd.DataFrame(data)
        dataframe.to_csv(embedd_file,index = False)