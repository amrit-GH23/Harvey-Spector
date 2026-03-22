# ⚖️ Jolly LLB — Indian Constitution RAG Assistant

A local **AI legal assistant** that answers natural‑language questions about the **Constitution of India (Articles 1–35)** using a **Retrieval‑Augmented Generation (RAG)** pipeline. Responses are grounded in constitutional text and include citations.

> **Note**: This project runs fully locally using **Docker + Ollama** (no external APIs required).

## 🚀 Features

- 🔎 Semantic search over Constitution Articles **1–35**
- ⚖️ Grounded legal answers with article references
- 🧠 Local LLM inference with **Ollama**
- 📚 Vector search with **ChromaDB**
- ⚡ **FastAPI** backend API
- 💻 **React** frontend
- 🐳 Dockerized LLM + vector database
- 🧩 **LangChain**-based RAG pipeline
- 🔒 Fully local (no cloud dependencies)
- 🎯 **Out-of-scope detection** — gracefully handles irrelevant queries

## 🧠 How it works

1. **Data ingestion**
   - Loads articles from `COI.json`
   - Chunks and embeds the text
2. **Vector storage**
   - Stores embeddings in **ChromaDB**
3. **Query processing**
   - Embeds the user query
   - Retrieves the most relevant articles
   - **Cross-encoder reranking** scores relevance
   - Out-of-scope queries are detected and handled gracefully
4. **Answer generation**
   - Sends retrieved context to the Ollama LLM
   - Returns an answer with citations

## 🏗️ System architecture

```text
User (React Frontend)
        |
        v
FastAPI Backend
        |
        v
LangChain RAG Pipeline
        |
        v
ChromaDB Vector Search
        |
        v
Relevant Articles Retrieved
        |
        v
Cross-Encoder Reranker (Out-of-scope detection)
        |
        v
Ollama LLM (llama3.1:8b)
        |
        v
Legal Response with Citations
```

## 🛠️ Tech stack

| Layer | Technology |
|---|---|
| Frontend | React |
| Backend | FastAPI |
| LLM Framework | LangChain |
| Local LLM | Ollama |
| Vector Database | ChromaDB |
| Containerization | Docker |
| Language | Python |

## 📂 Project structure

```text
Jolly-LLB/
├── .gitignore
├── frontend/               # React UI
├── backend/                # server
├── docker-compose.yml
├──Readme.md

```

## ⚙️ Prerequisites

- Python **3.10+**
- Node.js
- Docker
- Docker Compose (optional)

## Manual setup
## 🐳 Start Ollama (Docker) 

```bash
docker run -d \
  -p 11434:11434 \
  -v ollama:/root/.ollama \
  --name ollama \
  ollama/ollama
```

Pull the model:

```bash
docker exec -it ollama ollama pull llama3.1:8b
```

Verify:

```bash
docker exec -it ollama ollama list
```

## 🐳 Start ChromaDB

```bash
docker run -d \
  -p 8000:8000 \
  --name chromadb \
  chromadb/chroma
```

## 📦 Backend setup

Create and activate a virtual environment:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Linux / macOS:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 📚 Ingest constitution articles

Run ingestion once to embed the dataset and store it in ChromaDB:

```bash
python -m app.ingest
```

### ▶️ Run backend API

```bash
uvicorn app.main:app --reload --port 8080
```

API docs:

- http://localhost:8080/docs

## 💻 Frontend setup

```bash
cd frontend
npm install
npm run dev
```

Frontend runs on:

- http://localhost:5173

## 🧪 Example query

**Request** (`POST /query`):

```json
{
  "query": "What does the constitution say about equality?"
}
```

**Response**:

```json
{
  "answer": "Article 14 guarantees equality before the law and equal protection of the laws within the territory of India.",
  "sources": [
    {
      "article_no": "14",
      "title": "Equality before law",
      "part": "Part III - Fundamental Rights"
    }
  ]
}
```

## 🎯 Use cases

- Legal research assistants
- Law student study tools
- Civic education platforms
- AI-powered legal search
- Constitutional question answering

## ⚠️ Limitations

- Does not include case law or legal interpretation
- Not a substitute for professional legal advice

## 🔮 Future improvements

- Add the entire Constitution dataset
- Case law retrieval
- Better ranking of legal citations
- Streaming responses
- Authentication + deployment
- Multi-document legal RAG

## 📜 License

MIT License
