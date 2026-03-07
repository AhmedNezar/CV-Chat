import streamlit as st
import fitz 
import re
import io
from uuid import uuid4
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client.http.models import Distance, VectorParams

from config import model, client, collection_name, embedding

class ResumeMetadata(BaseModel):
    name: str = Field(description="Full name of candidate")

def extract_name(cv):
    structured_llm = model.with_structured_output(ResumeMetadata)
    prompt = f"Extract candidate full name from resume header. If not found return Unknown.\n\n{cv}"
    try:
        response = structured_llm.invoke(prompt)
        return response.name
    except Exception as e:
        print(f"Error extracting name: {e}")
        return "Unknown"

@st.cache_resource
def index_uploaded_documents(files):
    if len(files) != 5:
        st.sidebar.warning(f"Please upload exactly 5 files. Current: {len(files)}")
        return None

    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=3072, distance=Distance.COSINE)
    )

    client.create_payload_index(
        collection_name=collection_name,
        field_name="metadata.source",
        field_schema="keyword"
    )
    client.create_payload_index(
        collection_name=collection_name,
        field_name="metadata.name",
        field_schema="keyword"
    )

    st.info("Processing uploaded CVs...")
    cvs = []

    for i, uploaded_file in enumerate(files):
        # print(f"Processing {i+1} file.....")
        file_bytes = uploaded_file.read()
        pdf_stream = io.BytesIO(file_bytes)
        
        doc = fitz.open(stream=pdf_stream, filetype="pdf")
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        
        lines = full_text.split("\n")
        clean_text = "\n".join([re.sub(r"_{3,}", "", line.strip()) for line in lines])
        
        candidate_name = extract_name(clean_text[:1000])

        cvs.append(
            Document(
                page_content=clean_text,
                metadata={
                    "source": uploaded_file.name,
                    "name": candidate_name
                }
            )
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        add_start_index=True
    )
    splits = splitter.split_documents(cvs)
    
    vector_store = QdrantVectorStore(
        client=client,
        embedding=embedding,
        retrieval_mode=RetrievalMode.DENSE,
        collection_name=collection_name,
    )

    uuids = [str(uuid4()) for _ in splits]
    vector_store.add_documents(documents=splits, ids=uuids)
    st.success(f"Successfully indexed 5 candidates ({len(splits)} chunks).")
    return vector_store
