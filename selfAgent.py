import psycopg
import os
import uuid
import asyncio
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage , SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Command , interrupt
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg import connect
from psycopg.rows import dict_row
from typing import TypedDict , List, Annotated , Sequence, Literal , Dict
from pydantic import BaseModel, Field
from rag import hybrid_search , chunks
from mcp_tool import tavily_search


retriever = hybrid_search(chunks)

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

    doc = retriever.invoke(query)

    if not doc:
        print("No Revelant Document fetched")
    
      # job 2, runs per question
    return [d.page_content for d in doc]


class AgentState(TypedDict):

    query : str
    original_query : str
    document : List[str]
    is_websearch_needed : bool
    generation : str
    filter_documents : List
    unfilter_documents : List
    retry_count: int
    grounded : bool
    web_search_done : bool
    web_search_cancelled : bool 


class grade_schema(BaseModel):
    binary_score : Literal["yes", "no"] = Field(..., description="yes or no , wheather given document is revelant to question or not")


llm = ChatGroq(
    model = "openai/gpt-oss-120b",
    groq_api_key= GROQ_API_KEY,
)


grader_llm = llm.with_structured_output(grade_schema)


def retriever_node(state : AgentState) -> AgentState:

    query = state["query"]
    docs =  retriever_tool.invoke(query)
    

    return {"document":docs }



def grader(state : AgentState) -> dict:

    query = state["query"]
    document = state["document"]

    system  = """
            You are a grader evaluating whether a retrieved document is relevant to the user's question.

            Compare the question with the retrieved document.

            - Return "yes" if the document contains information that is relevant and useful for answering the question.
            - Return "no" if the document is unrelated or does not contain useful information for answering the question.

            Be strict: only return "yes" when the document provides meaningful information related to the question.
            """

    grader_prompt = ChatPromptTemplate.from_messages([
       ("system", system),
       ("human", "query: {query}\n\ndocument: {document}")
    ])

    chain = grader_prompt | grader_llm

    filtered_document = []
    unfiltered_document = []

    for doc in document:

        response =  chain.invoke({
            "query":query,
            "document": doc
        })

        if response.binary_score == "yes":
            filtered_document.append(doc)
        elif response.binary_score == "no":
            unfiltered_document.append(doc)

    if filtered_document:
        web_search = False
    else:
        web_search = True
    
    
    return {
        "filter_documents":filtered_document,
        "unfilter_documents": unfiltered_document,
        "query": query,
        "is_websearch_needed" : web_search
    }


def grader_should_continue(state : AgentState):

    web_search = state["is_websearch_needed"]

    if web_search:
        return "rewrite_query" 
    else:
        return "generator"


def rewrite_query(state :AgentState ) -> Dict:

    query = state["query"]
    retry_conut =  state.get('retry_count', 0) + 1   

    system = """
        You are a query rewriter for a search system. The user's question did not retrieve
        useful documents, so rewrite it to get better search results.

        Rules:
        - Keep the original meaning and intent. Do not change what is being asked.
        - Make the question more specific and clear.
        - Replace vague words with precise keywords, and spell out abbreviations.
        - Add closely related terms or synonyms that a relevant document might use.
        - Remove filler words and anything that isn't needed for searching.
        - Do NOT answer the question and do NOT add facts that the question doesn't imply.

        Return ONLY the rewritten question as one line. No explanation, no quotes.
        """

    
    rewriter_prompt = ChatPromptTemplate.from_messages([
        ("system" , system),
        ("human" , "Original question: {query} ")
    ])

    rewriter_chain  = rewriter_prompt | llm | StrOutputParser()

    response =  rewriter_chain.invoke({
        "query" : query
    })

    return {
        "query" : response,
        "retry_count" : retry_conut
    }


def rewriter_should_continue(state : AgentState) -> Dict:

    retry_count = state["retry_count"]

    if retry_count >= 2 :
        return "web_search"
    else:
        return "retriever"


def generator_node(state : AgentState)  -> Dict:


    query = state["original_query"]
    context = "\n\n".join(state["filter_documents"])

    system = """
        You are a helpful assistant that answers questions using only the provided context.

        Rules:
        - Answer using ONLY the information in the context.
        - If the context does not contain enough information, say "I don't have enough information to answer that." Do not guess.
        - Do not use outside knowledge and do not make up facts.
        - Be clear and concise. Use bullet points only when they make the answer easier to read.
        - If the context has several pieces of information, combine them into one answer.
        """

    generator_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("user", "\n\n Query : {query} \n\n context : {context}")
    ])

    generator_chain =  generator_prompt | llm | StrOutputParser()

    response =  generator_chain.invoke({
        "context" : context,
        "query" : query
    })

    return {
        "generation" : response,
    }

class check_schema(BaseModel):
    is_grounded: Literal["yes", "no"] = Field(..., description="yes if every claim in the answer is supported by the context")
    answers_question: Literal["yes", "no"] = Field(..., description="yes if the answer actually addresses the question")

check_llm =  llm.with_structured_output(check_schema)


def check_answer(state : AgentState) -> Dict:

    query = state["original_query"]
    generated_answer  = state["generation"]

    system = """
            You are a strict checker. You are given a context, a question, and an answer.
            Do two checks:

            1. is_grounded: Is every claim in the answer supported by the context?
            Return "no" if the answer contains any fact, number, or detail that is not in the context.

            2. answers_question: Does the answer actually address what the question asks?
            Return "no" if the answer is off-topic, incomplete, or says it doesn't have enough information.

            Be strict. Judge only from the context given. Do not use your own knowledge.
            """

    check_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("user", "here's the full details: \n question : {query} \n\n context : {context} \n\n answer : {answer}  ")
    ])

    
    check_answer_chain = check_prompt | check_llm

    response =  check_answer_chain.invoke({
        "query" : query,
        "context" : "\n\n".join(state["filter_documents"]),
        "answer": generated_answer
    })

    if response.is_grounded == "yes".strip().lower() and response.answers_question == "yes".strip().lower():
        grounded = True
    else:
        grounded = False
    

    return {
        "grounded" : grounded
    }

def check_should_continue(state: AgentState):
    if state["grounded"] or state.get("web_search_done", False):
        return "End"
    return "web_search"

def webSearch(state : AgentState) :

    query = state["query"]

    decision = interrupt({
        "action" : "Web Search",
        "query" : query,
        "approval" : "Allow web Search?"
    })

    if str(decision).strip().lower() != "y":
        update = {"web_search_done": True, "web_search_cancelled": True}
        if not state.get("generation"):
            update["generation"] = "Web search cancelled by user."
        return update


    response = asyncio.run(tavily_search(query))

    return {
        "filter_documents": response,
        "web_search_done": True,
        "web_search_cancelled": False,  
    }


def websearch_should_continue(state: AgentState):
    if state.get("web_search_cancelled", False):
        return "End"
    return "generator"

graph = StateGraph(AgentState)

graph.add_node("retriever" , retriever_node)
graph.add_node("grader", grader)
graph.add_node("rewriter", rewrite_query)
graph.add_node("websearcher" , webSearch)
graph.add_node("generator", generator_node)
graph.add_node("checker", check_answer)


graph.add_edge(START , "retriever")
graph.add_edge("retriever", "grader")
graph.add_conditional_edges(
    "grader",
    grader_should_continue,
    {
        "generator" : "generator",
        "rewrite_query" : "rewriter"
    }
)
graph.add_conditional_edges(
    "rewriter",
    rewriter_should_continue,
    {
        "web_search" : "websearcher",
        "retriever" : "retriever",
    }
)
graph.add_conditional_edges(
    "websearcher",
    websearch_should_continue,
    {
        "End": END,
        "generator": "generator"
    }
)
graph.add_edge("generator" , "checker")
graph.add_conditional_edges(
    "checker",
    check_should_continue,
    {
        "End" : END,
        "web_search" : "websearcher"
    }
)


database_url =  get_database_url()

conn =  psycopg.connect(
    database_url,
    row_factory= dict_row,
    autocommit=True
)

checkpointer = PostgresSaver(
    conn
)
checkpointer.setup()


HybridRag = graph.compile(checkpointer=checkpointer)


if __name__ == "__main__":

    while True:  # OUTER loop: one pass per question

        user_query = input("\nAsk your question ('exit' to quit): ").strip()
        if user_query.lower() in {"exit", "quit", "q"}:
            break
        if not user_query:
            continue

        inputs = {
            "query": user_query,
            "original_query": user_query,
            "document": [],
            "generation": "",
            "filter_documents": [],
            "unfilter_documents": [],
            "retry_count": 0,
            "grounded": False,
            "web_search_done": False,
            "web_search_cancelled": False,
        }

        config = {"configurable": {"thread_id": str(uuid.uuid4())}}

        while True:  # INNER loop: only for the approval pause

            result = HybridRag.invoke(inputs, config=config)

            if "__interrupt__" not in result:
                print("\n" + result["generation"])
                break  # exits INNER loop only, so the outer loop asks the next question

            payload = result["__interrupt__"][0].value
            print(f"\nApproval for web search: {payload}")
            answer = input("Y/N Allow web search? ")
            inputs = Command(resume=answer)

        









    

        
    





    


















