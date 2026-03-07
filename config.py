import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_ollama import ChatOllama
from qdrant_client import QdrantClient

load_dotenv()

collection_name = os.getenv("QDRANT_COLLECTION")

embedding = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
# model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
model = ChatOllama(model="llama3.1:8b")

@st.cache_resource
def get_qdrant_client():
    qdrant_url = os.getenv("QDRANT_ENDPOINT")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    return QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

client = get_qdrant_client()
