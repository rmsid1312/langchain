from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from tavily import TavilyClient
tavily = TavilyClient()

@tool
def search(query: str) -> str:
    """
    Tool that searches over internet
    Args:
        query: The query to search for
    Returns:
        The search result
    """
    print(f"Search for {query}")
    return tavily.search(query=query)

llm = ChatOllama(temperature=0, model="qwen3:8b")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)

def main() -> None:
    result = agent.invoke({"messages":HumanMessage(content="search for 3 job postings for an ai enginner using langchain in the bay area on linkedin and list ther details")})
    print(result)

if __name__ == '__main__':
    main()