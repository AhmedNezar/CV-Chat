import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, Filter, FieldCondition, MatchText
from uuid import uuid4
from pydantic import BaseModel, Field
import fitz 
import os
import re
import io


st.set_page_config(page_title="CV AI Recruiter", layout="wide")
st.title("📄 AI CV Recruitment Assistant")

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
collection_name = os.getenv("QDRANT_COLLECTION")

embedding = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
# model = ChatOllama(model="llama3.1:8b")


st.sidebar.title("Resume Upload")
uploaded_files = st.sidebar.file_uploader(
    "Upload exactly 5 Candidate CVs (PDF)", 
    type="pdf", 
    accept_multiple_files=True
)

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
def get_qdrant_client():
    qdrant_url = os.getenv("QDRANT_ENDPOINT")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    return QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

client = get_qdrant_client()


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

if uploaded_files and len(uploaded_files) == 5:
    vector_store = index_uploaded_documents(uploaded_files)
elif uploaded_files and len(uploaded_files) != 5:
    st.sidebar.error("Upload status: Waiting for exactly 5 files.")


@tool
def retrieve_grouped_context(query: str):
    """Retrieve candidate CV context for multiple candidates"""
    embedded_query = embedding.embed_query(query)

    response = client.query_points_groups(
        collection_name=collection_name,
        query=embedded_query,
        group_by="metadata.source",
        limit=4,
        group_size=4
    )

    context = []

    for group in response.groups:
        content = [hit.payload["page_content"] for hit in group.hits if hit.score > 0.65]
        metadata = group.hits[0].payload["metadata"]

        if len(content) > 0:
            context.append(
                f"Candidate: {metadata['name']}\n"
                f"Context:\n{"\n".join(content)}\n"
                f"------------------"
            )

    return "\n".join(context)

@tool
def retrieve_candidate_context(query: str, candidate_name: str):
    """Retrieve detailed CV context for a specific candidate"""

    embedded_query = embedding.embed_query(query)

    search_filter = Filter(
        must=[
            FieldCondition(
                key="metadata.name",
                match=MatchText(text=candidate_name)
            )
        ]
    )

    results = client.query_points(
        collection_name=collection_name,
        query=embedded_query,
        query_filter=search_filter,
        limit=5
    )

    context = []

    for point in results.points:
        payload = point.payload
        content = payload["page_content"]
        metadata = payload["metadata"]

        context.append(
            f"Candidate: {metadata['name']}\n"
            f"Context:\n{content}\n"
            f"------------------"
        )

    return "\n".join(context)


system_prompt = """You are an expert AI Recruitment Assistant.
Your goal is to help users find, rank, compare, and analyze candidates based on their resumes.

--------------------------------------------------

### TOOL USAGE

You have access to retrieval tools.

- If the user asks about candidate skills, experience, ranking, comparison, or suitability for a role, you MUST retrieve relevant candidate data before answering.
- If the question involves multiple candidates (e.g., "Who is best for data analyst?"), use grouped retrieval.
- If the question involves one named candidate, use single-candidate retrieval.
- Only ask for clarification if the user’s request is truly ambiguous and cannot be reasonably inferred.
- When converting user queries to tool inputs, do not remove any keywords, skills, or technologies mentioned by the user.

--------------------------------------------------


### ROLE MATCHING & VALIDATION LOGIC

1. **Verify Exact Role Integrity**: 
   - Before evaluating candidates, compare the user's requested role (e.g., "Ai teams engineer") against the actual titles and primary experience listed in the retrieved CVs.
   - You must treat unique keywords (like "Team", "Lead", "Manager", or "Staff") as mandatory semantic requirements. 
   - **Crucial Rule**: "AI Engineer" is NOT the same as "AI Team Engineer". If the retrieved CVs only contain "AI Engineer", you must declare a mismatch.

2. **Refusal Mechanism for Imaginary Roles**:
   - If the specific role title or key modifier (e.g., "teams") does not appear in the candidate's history or if the role seems "imaginary" relative to the context, you MUST NOT recommend any candidate.
   - Instead, respond exactly like this: "I found candidates for [Existing Role A] and [Existing Role B], but I did not find any candidates specifically for '[User's Requested Role]'. Would you like to see the closest matches instead?"

If the user asks:
- "Who is the best candidate for X?"
- "Who fits a Data Analyst role?"
- "Rank candidates for Machine Learning Engineer"

You MUST:

1. Retrieve multiple candidates.
2. Infer the typical requirements of that role.
3. Evaluate each candidate based only on their CV content.
4. Rank them objectively.
5. Clearly explain why the top candidate is most suitable.

--------------------------------------------------

### RESPONSE GUIDELINES

1. Always mention candidate names.
2. Only use retrieved CV information.
3. If ranking candidates, provide structured comparison (table or bullet points).
4. Do not invent skills.
5. If no candidate clearly fits, say so and explain why.
6. Only mention skills, experience, or tools that are explicitly listed in the retrieved context.
7. Do not assume any variant or similar skill. For example, C is not C++ and vice versa.
8. If the candidate does not have all requested skills, clearly state which are missing.

--------------------------------------------------

### DATA FORMAT FROM TOOLS

Tool output format:

Candidate: [Name]
Context:
[Content]
------------------

Use only this information when analyzing candidates.

--------------------------------------------------

### TONE

Professional, analytical, decisive, and confident.
Do not default to asking for more details when a reasonable evaluation can be made.
"""


agent = create_agent(
    model=model,
    tools=[retrieve_grouped_context, retrieve_candidate_context],
    system_prompt=system_prompt
)


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about candidates..."):

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        for event in agent.stream(
            {"messages": [{"role": "user", "content": prompt}]},
            stream_mode="values"
        ):

            last_msg = event["messages"][-1]

            if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
                print("\n" + "="*40)
                print(f"🛠️ AGENT INITIATED TOOL CALL")
                for tc in last_msg.tool_calls:
                    print(f"Tool Name: {tc['name']}")
                    print(f"Search Query: {tc['args']}")
                print("="*40 + "\n")


            elif isinstance(last_msg, ToolMessage):
                print(f"✅ TOOL RETURNED DATA: {last_msg.content}\n")


            elif isinstance(last_msg, AIMessage) and last_msg.content:
                content = last_msg.content
                
                if isinstance(content, list):
                    full_response = "".join([c['text'] for c in content if c.get('type') == 'text'])
                else:
                    full_response = content
                
                response_placeholder.markdown(full_response)

        if full_response:
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )
