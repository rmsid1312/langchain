import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
load_dotenv()


def main() -> None:

    information="""Born into the family that controls the Samsung conglomerate,[3] Lee has held various positions in the company since joining Samsung in 1991. Lee has consistently ranked among the wealthiest persons in South Korea, with a net worth reaching tens of billions of dollars.[4][5] He has also been named to Forbes’ list of the world’s top 50 billionaries.[6]"""

    summary_template="""
    given the information {information} about a person I want you to create.
    1. A short summary
    2. two interesting facts about them
    """

    system_prompt_template = PromptTemplate(input_variables=["information"], template=summary_template)
    llm = ChatOllama(temperature=0, model="qwen3:8b")
    chain = system_prompt_template | llm
    response = chain.invoke(input={"information": information})
    print(response.content)

if __name__ == "__main__":
    main()
