import streamlit as st
from langchain_core.messages import AIMessage, ToolMessage

# Initialize page config before importing other modules that might use st functions
st.set_page_config(page_title="CV AI Recruiter", layout="wide")

from indexing import index_uploaded_documents
from agent import agent

st.title("📄 AI CV Recruitment Assistant")

st.sidebar.title("Resume Upload")
uploaded_files = st.sidebar.file_uploader(
    "Upload exactly 5 Candidate CVs (PDF)", 
    type="pdf", 
    accept_multiple_files=True
)

if uploaded_files and len(uploaded_files) == 5:
    vector_store = index_uploaded_documents(uploaded_files)
elif uploaded_files and len(uploaded_files) != 5:
    st.sidebar.error("Upload status: Waiting for exactly 5 files.")

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
