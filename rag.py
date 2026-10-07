
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
print(COHERE_API_KEY)

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

    all_pages = []
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












