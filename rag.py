
from chromadb.api.types import validate_embedding_function
from langchain_community import vectorstores
import os
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.document_loaders import  WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import InMemoryStore
from langchain_cohere import CohereRerank
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.vectorstores import Chroma


load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")


def pdf_loader(path: str):

    if not os.path.exists(path):
        raise FileNotFoundError("File not found error, please give  valid path")
    
    loader = PyPDFLoader(path)

    try:
        doc = loader.load()
        print(f"Length of pages {len(doc)}")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 150
        )

        splitted_chunks = splitter.split_documents(doc)
        print(f"Length of splitted chunks : {len(splitted_chunks)}")
    except Exception as e:
        raise ValueError("Error while loading pdf, give valid file path")

    
    return splitted_chunks

def web_loader(url : str):

    loader = WebBaseLoader(url)

    try:
        doc = loader.load()
        print(f"Length of pages {len(doc)}")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 150
        )
        splitted_chunks = splitter.split_documents(doc)
        print(f"Length of splitted chunks : {len(splitted_chunks)}")
    
    except Exception as e:
        raise ValueError("Erro while loading web {e}, please enter valid url ")

    
    return splitted_chunks



selected_type = input("Select the document type 'url' or 'pdf :  ").strip().lower()

if selected_type  == "pdf":
    path = input("Paste your file path : ").strip().strip('"').strip("'")
    chunks = pdf_loader(path)
elif selected_type == "url":
    url = input("paste your url : ").strip()
    chunks = web_loader(url)
else:
    raise ValueError(f"Invalid document type: '{selected_type}'. Must be 'pdf' or 'url'.")



def hybrid_search(chunks : str):

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3"
    )

    vectorstore = Chroma.from_documents(
        embedding = embeddings,
        documents=chunks,
        persist_directory="./chroma_pdr",
        collection_name="hybrid_demo",
    )

    vector_search = vectorstore.as_retriever(
        search_type = "mmr",
        search_kwargs={
            "k": 10,
            "fetch_k": 15,
            "lambda_mult": 0.25
        }
    )

    bm25_search = BM25Retriever.from_documents(
        documents=chunks,
        k = 10
    )

    ensemble = EnsembleRetriever(
        retrievers=[vector_search, bm25_search],
        weights=[0.7, 0.3]
    )

    reranker = CohereRerank(
        model= "rerank-v4.0-pro",
        top_n = 5,
        cohere_api_key=COHERE_API_KEY,
    )

    retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=ensemble
    )

    return retriever


if __name__ == "__main__":
    selected_type = input("Select the document type 'url' or 'pdf :  ").strip().lower()

    if selected_type  == "pdf":
        path = input("Paste your file path : ").strip().strip('"').strip("'")
        chunks = pdf_loader(path)
    elif selected_type == "url":
        url = input("paste your url : ").strip()
        chunks = web_loader(url)
    else:
        raise ValueError(f"Invalid document type: '{selected_type}'. Must be 'pdf' or 'url'.")


    retriever = hybrid_search(chunks=chunks)

    while True:
        query = input("\nAsk a question (or 'exit'): ").strip()
        if query.lower() in {"exit", "quit", "q"}:
            break

        results = retriever.invoke(query)

        for i, doc in enumerate(results, 1):
            score = doc.metadata.get("relevance_score")
            print(f"\n--- Result {i} (score: {score}) ---")
            print(doc.page_content[:500])
            print("Source:", doc.metadata.get("source"), "| Page:", doc.metadata.get("page"))



















