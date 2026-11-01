# AI-Powered Knowledge Base Search with Enrichment

## What it does

This is a retrieval-augmented generation (RAG) system that searches
documents you upload, answers questions from them with a confidence
score and a completeness flag, and, when the answer is incomplete or
low-confidence, automatically enriches it from trusted external
sources (Wikipedia, arXiv, and PubMed, in that priority order).
Uploads are PDF, TXT, and DOCX; search is vector-based using
ChromaDB; answers are generated with GPT-4 and returned as
structured JSON with answer, confidence, missing information, and
sources. There is also a 1-5 star rating system for answers, and a
web UI served by a FastAPI backend.

## 📁 Project Structure Notice

All functional code lives under a single main folder, and it has
not been moved in this update:

- **`RAG_1/`** → Contains the complete codebase for the AI-Powered
  Knowledge Base (RAG system).
  - Includes FastAPI backend, data processing, vector search, and
    frontend UI.
  - All setup and run scripts (`setup.sh`, `run.sh`) are located
    inside this folder.
  - Full documentation lives inside `RAG_1/` as well, starting
    with `RAG_1/README.md` and `RAG_1/INDEX.md`.

## ⚙️ How to Run

```bash
# Clone this repository (not RAG_1 on its own)
git clone https://github.com/santhoshchakilamcs/AI-Powered-Knowledge-Base-Search-with-Enrichment.git
cd AI-Powered-Knowledge-Base-Search-with-Enrichment/RAG_1

# Set up (installs dependencies; see RAG_1/README.md for details)
chmod +x setup.sh
./setup.sh

# Configure environment
cp .env.example .env
# Edit .env and add your OpenAI API key:
# OPENAI_API_KEY=sk-your-key-here

# Run the application
bash ./run.sh
# Open http://localhost:8000 in your browser
```

Requirements, stated honestly: Python 3.9 or higher, an OpenAI API
key (used for GPT-4 answers and `text-embedding-3-small`
embeddings), and about 2 GB free disk space, per `RAG_1/README.md`.
Auto-enrichment makes external requests to Wikipedia, arXiv, and
PubMed only when an answer is incomplete or below the confidence
threshold. There is no user authentication in this version; it is a
single-user system.
