from aiohttp import base_protocol
import os
import asyncio
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    raise ValueError("Invalid API key, please give valid API key in .env")

async def tavily_search(query):

    client = MultiServerMCPClient({
        "tavily-remote-mcp": {
            "transport":"streamable_http",
            "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY}"
            }
        }
    )

    tools =  await client.get_tools()


    tool_map = {tool.name: tool for tool in tools}
    
    tavily_search = tool_map["tavily_search"]
    

    result = await tavily_search.ainvoke({
    "query": query
})

    structured_result = {
        "query": query,
        "results": []
    }

    for block in result:
        if block["type"] == "text":
            structured_result["results"].append(block["text"])

    print(structured_result)
    
if __name__ == "__main__":
    asyncio.run(tavily_search())
