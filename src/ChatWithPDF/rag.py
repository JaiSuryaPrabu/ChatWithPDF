# this python file contains all steps from the retrieval to generation code
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer,util
from transformers import AutoTokenizer , AutoModelForCausalLM


class RAG:

    def __init__(self):
        self.model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.embedding_model_name = "all-mpnet-base-v2"
        self.embeddings_filename = "embeddings.csv"
        self.data_pd = pd.read_csv(self.embeddings_filename)
        self.data_dict = pd.read_csv(self.embeddings_filename).to_dict(orient='records')
        self.data_embeddings = self.get_embeddings()
        
        self.embedding_model = SentenceTransformer(model_name_or_path = self.embedding_model_name,device = self.device)
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        
        # LLM
        self.llm_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=self.model_id,
                                                         torch_dtype=torch.float16).to(self.device)

    def get_embeddings(self) -> list:
        """Returns the embeddings from the csv file"""
        data_embeddings = []
    
        for tensor_str in self.data_pd["embeddings"]:
            values_str = tensor_str.split("[")[1].split("]")[0]
            values_list = [float(val) for val in values_str.split(",")]
            tensor_result = torch.tensor(values_list)
            data_embeddings.append(tensor_result)
    
        data_embeddings = torch.stack(data_embeddings).to(self.device)
        return data_embeddings


    def retrieve_relevant_resource(self,user_query : str , k = 5):
        """Function to retrieve relevant resource"""
        query_embedding = self.embedding_model.encode(user_query, convert_to_tensor = True).to(self.device)
        dot_score = util.dot_score( a = query_embedding, b = self.data_embeddings)[0]
        score , idx = torch.topk(dot_score,k=k)
        return score,idx

    def prompt_formatter(self,query: str, context_items: list[dict]) -> str:
        """
        Augments query with text-based context from context_items.
        """
        # Join context items into one dotted paragraph
        context = "- " + "\n- ".join([item["sentence_chunk"] for item in context_items])
    
        base_prompt = """Based on the following context items, please answer the query.
    Use the following example as reference for the ideal answer style.
    \nExample :
    Query: What are the fat-soluble vitamins?
    Answer: The fat-soluble vitamins include Vitamin A, Vitamin D, Vitamin E, and Vitamin K. These vitamins are absorbed along with fats in the diet and can be stored in the body's fatty tissue and liver for later use. Vitamin A is important for vision, immune function, and skin health. Vitamin D plays a critical role in calcium absorption and bone health. Vitamin E acts as an antioxidant, protecting cells from damage. Vitamin K is essential for blood clotting and bone metabolism.
    \nNow use the following context items to answer the user query:
    {context}
    \nRelevant passages: <extract relevant passages from the context here>
    User query: {query}
    Answer:"""
    
        # Update base prompt with context items and query   
        base_prompt = base_prompt.format(context=context, query=query)
    
        # Create prompt template for instruction-tuned model
        dialogue_template = [
            {"role": "user",
            "content": base_prompt}
        ]
    
        # Apply the chat template
        prompt = self.tokenizer.apply_chat_template(conversation=dialogue_template,
                                              tokenize=False,
                                              add_generation_prompt=True)
        return prompt

    def query(self,user_text : str):
        scores, indices = self.retrieve_relevant_resource(user_text)
        context_items = [self.data_dict[i] for i in indices]
        prompt = self.prompt_formatter(query=user_text,context_items=context_items)
        input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.llm_model.generate(**input_ids,max_new_tokens=256) 
        output_text = self.tokenizer.decode(outputs[0])
        output_text = output_text.split("<|assistant|>")
        output_text = output_text[1].split("</s>")[0]
        
        return output_text

    