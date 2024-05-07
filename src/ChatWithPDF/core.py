from embeddings import Embeddings
from rag import RAG

rag_ = None

def process_pdf(file:str):
    emb = Embeddings(file)
    emb.save_the_embeddings()
    global rag_
    rag_ = RAG()

def process_query(user_text:str):
    global rag_
    return rag_.query(user_text)