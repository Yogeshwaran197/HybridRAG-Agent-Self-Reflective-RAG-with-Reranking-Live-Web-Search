import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Command , Interrupt
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg import Connection
from psycopg.rows import dict_row
from typing import TypedDict , List, Annotated , Sequence, Literal
from pydantic import BaseModel, Field
from rag import hybrid_search


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


@tool
def retriever_tool(query : str):
    """Search and return relevant document chunks from the indexed blog posts about LLM agents, prompt engineering, and adversarial attacks on LLMs. Use this when the question relates to those topics."""

    retriever = hybrid_search()
    doc = retriever.invoke(query)

    if not doc:
        print("No Revelant Document fetched")

    result = []

    for d in doc:
        result.append(f"Document : {d.page_content}")
    
    return "\n\n".join(result)

class AgentState(TypedDict):

    query : str
    document : Annotated[Sequence[str], add_messages]
    is_websearch_needed : bool
    generation : str
    filter_documents : List
    unfilter_documents : List
    retry_count: int


class grade_schema(BaseModel):
    binary_score : Literal["yes", "no"] = Field(..., description="yes or no , wheather given document is revelant to question or not")


llm = ChatGroq(
    model = "qwen/qwen3.8-27b",
    groq_api_key= GROQ_API_KEY,
)




















