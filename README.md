# PRESENTATION VIDEO LINK

https://youtu.be/aZBWIjpJ-3s

# AI-Powered-Document-Assistant

AI-Powered-Document-Assistant is a full-stack local RAG web application that lets users ask questions about documents stored on their computer.

Instead of relying on a static FAQ or manual keyword search, the app indexes local files, retrieves the most relevant passages with vector search and reranking, and uses a locally hosted Ollama model to generate grounded answers with source references.

## Features

- Ask natural language questions about local documents
- Build a searchable knowledge base from a folder on your machine
- Retrieval-Augmented Generation (RAG) pipeline
- Semantic embeddings for document search
- FAISS vector storage for fast similarity lookup
- Cross-encoder reranking for improved retrieval quality
- Source citations included with responses
- Automatic refresh if indexed source files change
- Evaluation endpoint for batch question testing
- Full-stack architecture
  - Python FastAPI backend
  - HTML / CSS / JavaScript frontend

## Tech Stack

### Frontend

- HTML
- CSS
- JavaScript

### Backend

- Python
- FastAPI
- Uvicorn
- Sentence Transformers
- FAISS
- NumPy
- PyPDF
- Requests
- Transformers
- Torch

### Local AI / Models

- Ollama
- `BAAI/bge-small-en-v1.5` for embeddings
- `cross-encoder/ms-marco-MiniLM-L-6-v2` for reranking
- `qwen2.5:3b` via Ollama for answer generation

## Installation & Setup

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd AI-Powered-Document-Assistant
```

### 2. Create Virtual Environment

Mac / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install and Start Ollama

Make sure Ollama is installed, then pull the model used by this project:

```bash
ollama pull qwen2.5:3b
```

Start Ollama:

```bash
ollama serve
```

### 5. Configure Environment Variables

This project already includes a `.env` file. The default configuration is:

```env
APP_NAME=Local AI Document Assistant
APP_VERSION=0.1.0
DATA_DIR=./data
CHUNK_SIZE=500
CHUNK_OVERLAP=100
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
TOP_K=5
MIN_SOURCE_SIMILARITY_SCORE=0.35
RETRIEVAL_CANDIDATES=15
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:3b
LOG_LEVEL=INFO
```

### 6. Run the Backend Server

```bash
python -m uvicorn app.main:app --reload
```

You should see the app running at:

```text
http://127.0.0.1:8000
```

### 7. Open the Web App

Open this URL in your browser:

```text
http://127.0.0.1:8000
```

## How to Use

1. Start Ollama
2. Start the FastAPI server
3. Open the web app in your browser
4. Enter a local folder path containing `.txt` or `.pdf` files
5. Click `Build Knowledge Base`
6. Ask a question about the indexed documents
7. Review the answer and source citations

Example folder to test with:

```text
sample_docs
```

## API Endpoints

- `GET /health` checks server status
- `POST /ingest` indexes a local folder
- `POST /query` asks a question against indexed documents
- `POST /evaluate` runs a batch evaluation from a JSON file

## Supported File Types

- PDF
- TXT

## Project Structure

```text
AI-Powered-Document-Assistant/
│
├── app/
│   ├── api.py               # FastAPI routes
│   ├── core.py              # Ingestion, retrieval, reranking, and RAG pipeline
│   ├── main.py              # FastAPI app entrypoint
│   ├── models.py            # Pydantic models and settings
│   └── static/
│       ├── app.js           # Frontend logic
│       ├── index.html       # Web interface
│       └── styles.css       # Styling
│
├── data/                    # Saved vector index and chunk metadata
├── sample_docs/             # Sample documents and evaluation files
├── .env                     # Environment configuration
├── requirements.txt         # Python dependencies
└── README.md
```

## Notes

- The current implementation works with local folders, not Google Drive
- Answers are grounded in indexed content, but output quality depends on the documents and local model
- The first run may take longer because embedding and reranking models need to be downloaded
