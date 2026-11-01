import os
from dotenv import load_dotenv
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_unstructured import UnstructuredLoader
from langchain_ollama import OllamaEmbeddings
load_dotenv()

if __name__ == '__main__':
    loader = UnstructuredLoader(file_path=r"C:\Users\rmsid\workspace\langchain\mediumblog1.txt", chunking_strategy="basic", max_characters=1000000)
    document = loader.load()

    print("splitting...")
    text_spliter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_spliter.split_documents(documents=document)
    print(f"created {len(texts)} chunks")

    embeddings = OllamaEmbeddings(model="bge-m3")
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    print("ingesting...")
    PineconeVectorStore.from_documents(
        texts, embedding=embeddings, index_name=os.environ["INDEX_NAME"]
    )
    print("finish")