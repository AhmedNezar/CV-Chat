# AI CV Recruitment Assistant 📄

A powerful Streamlit-based AI recruiting application that analyzes candidate resumes (PDFs), stores them in a **Qdrant** vector database, and uses **LangChain** with **Gemini** (or local **Ollama** models) to help recruiters find, rank, and compare candidates based on their CVs.

## Features ✨

- **Batch Resume Processing**: Upload exactly 5 candidate CVs (PDF) at once.
- **Automatic Text Extraction**: Uses PyMuPDF (`fitz`) to extract full text and extract candidate names via structured LLM output.
- **Smart Vector Storage**: Chunks CVs and embeds them using Google Generative AI Embeddings, storing them in a =remote Qdrant collection.
- **Agentic AI Assistant**: A custom LangChain Agent capable of:
  - Answering natural language questions about candidate skills and experience.
  - Using **Grouped Retrieval** to query multiple candidates simultaneously (e.g., "Who is best for a Data Analyst role?").
  - Using **Targeted Retrieval** for specific candidate deep-dives.
  - Validating requested roles against actual CV content to prevent hallucinated matches.

## Project Structure 📁

```text
CV Chat/
│
├── .env                     # Environment variables (API keys, endpoints)
├── config.py                # Initializes LLMs, Embeddings, and Qdrant connections
├── indexing.py              # PDF extraction, chunking, and Qdrant ingestion
├── tools.py                 # LangChain retrieval tools for the agent
├── agent.py                 # System prompt and LangChain Agent definition
└── app.py                   # Main Streamlit UI and chat interface
```

## Prerequisites 🛠️

- Python 3.9+
- [Qdrant](https://qdrant.tech/) instance (either local Docker or Qdrant Cloud)
- Google Gemini API Key (or an active Ollama service for local inferences)

## Installation & Setup 🚀

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd "CV Chat"
   ```

2. **Create a virtual environment & install dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
   *(Ensure you have packages like `streamlit`, `langchain`, `langchain-google-genai`, `qdrant-client`, `pymupdf`, `pydantic`, `python-dotenv` installed)*

3. **Configure Environment Variables:**
   Create a `.env` file in the root directory and add the following:
   ```env
   GOOGLE_API_KEY=your_gemini_api_key_here
   QDRANT_ENDPOINT=http://localhost:6333  # Or your Qdrant Cloud URL
   QDRANT_API_KEY=your_qdrant_api_key_here # Omit if using local unauthenticated Qdrant
   QDRANT_COLLECTION=candidate_resumes
   ```

4. **Run the Application:**
   ```bash
   streamlit run app.py
   ```

## Usage 💡

1. Open the application in your browser (usually `http://localhost:8501`).
2. Navigate to the sidebar and upload **exactly 5 PDF resumes**.
3. Wait for the success message indicating that the CVs have been indexed.
4. Use the chat interface to ask questions such as:
   - *"Which candidate has the most experience with machine learning?"*
   - *"Compare the top 2 candidates for a Data Engineering role."*
   - *"Does [Candidate Name] have experience with AWS?"*

## Technologies Used 💻

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Orchestration**: [LangChain](https://python.langchain.com/)
- **LLM & Embeddings**: [Google Gemini Pro / Flash](https://ai.google.dev/) (also supports local [Ollama](https://ollama.ai/))
- **Vector Database**: [Qdrant](https://qdrant.tech/)
- **PDF Extraction**: [PyMuPDF (fitz)](https://pymupdf.readthedocs.io/)
