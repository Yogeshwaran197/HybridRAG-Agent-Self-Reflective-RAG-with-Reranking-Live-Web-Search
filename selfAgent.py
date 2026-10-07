import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command , Interrupt
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg import Connection
from psycopg.rows import dict_row
from typing import TypedDict , List
from pydantic import BaseModel, Field


load_dotenv()

def get_database_url():
    DATABASE_URL =  os.getenv("DATABASE_URL")

    if not DATABASE_URL:
        raise ValueError("Invalid database url, need valid database url")

    
    if "sslmode" not in DATABASE_URL:
        seperator = "&" if  "?" in DATABASE_URL else "?"
        DATABASE_URL = f"{DATABASE_URL}{seperator}sslmode=require"

    return DATABASE_URL


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Error incorrect API key, please give correct API key")










