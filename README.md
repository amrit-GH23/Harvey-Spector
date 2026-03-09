⚖️ Harvey Spector — Indian Constitution RAG Assistant

Harvey Spector is a local AI legal assistant that allows users to query the Constitution of India (Articles 1–35) using natural language.

The system uses a Retrieval-Augmented Generation (RAG) architecture to retrieve relevant constitutional articles and generate context-aware legal explanations with citations.

Unlike generic LLM responses, answers are grounded in actual constitutional text, reducing hallucinations.

The entire system runs locally using Docker + Ollama, requiring no external APIs.

🚀 Features

🔎 Semantic Search over Constitution Articles (1–35)

⚖️ Grounded legal answers with article references

🧠 Local LLM inference using Ollama

📚 Vector search with ChromaDB

⚡ FastAPI backend API

💻 React frontend interface

🐳 Dockerized LLM + Vector DB

🧩 LangChain-based RAG pipeline

🔒 Fully local (no OpenAI or cloud dependencies)

🧠 How It Works

The system follows a Retrieval-Augmented Generation pipeline:

1️⃣ Data Ingestion

Constitution articles are loaded from COI.json

Articles are chunked and embedded

2️⃣ Vector Storage

Embeddings are stored in ChromaDB

3️⃣ Query Processing

User query is converted into embeddings

Similar constitutional articles are retrieved

4️⃣ Answer Generation

Retrieved articles are passed to the Ollama LLM

The model synthesizes a legally grounded response

🏗 System Architecture
User (React Frontend)
        │
        ▼
FastAPI Backend
        │
        ▼
LangChain RAG Pipeline
        │
        ▼
ChromaDB Vector Search
        │
        ▼
Relevant Articles Retrieved
        │
        ▼
Ollama LLM (llama3.1:8b)
        │
        ▼
Legal Response with Citations
🛠 Tech Stack
Layer	Technology
Frontend	React
Backend	FastAPI
LLM Framework	LangChain
Local LLM	Ollama
Vector Database	ChromaDB
Containerization	Docker
Language	Python
📂 Project Structure
Harvey-Spector/
│
├── COI.json                # Constitution Articles Dataset (1–35)
├── requirements.txt
├── .env
├── .gitignore
│
├── frontend/               # React UI
│
└── app/
    ├── config.py           # Environment configuration
    ├── ingest.py           # Data ingestion and embeddings
    ├── rag.py              # Retrieval + generation pipeline
    └── main.py             # FastAPI API server
⚙️ Prerequisites

Before running the project, install:

Python 3.10+

Node.js

Docker

Docker Compose (optional)

🐳 Start Ollama (Docker)

Run Ollama locally using Docker:

docker run -d \
-p 11434:11434 \
-v ollama:/root/.ollama \
--name ollama \
ollama/ollama

Pull the model:

docker exec -it ollama ollama pull llama3.1:8b

Verify:

docker exec -it ollama ollama list
🐳 Start ChromaDB

Using Docker:

docker run -d \
-p 8000:8000 \
--name chromadb \
chromadb/chroma
📦 Backend Setup

Create virtual environment:

python -m venv venv

Activate:

Windows

venv\Scripts\activate

Linux / Mac

source venv/bin/activate

Install dependencies:

pip install -r requirements.txt
📚 Ingest Constitution Articles

Run ingestion once to embed the dataset.

python -m app.ingest

This creates embeddings for Articles 1–35 and stores them in ChromaDB.

▶️ Run Backend API
uvicorn app.main:app --reload --port 8080

API docs available at:

http://localhost:8080/docs
💻 Frontend Setup

Navigate to frontend:

cd frontend

Install dependencies:

npm install

Run development server:

npm run dev

Frontend will run on:

http://localhost:5173
🧪 Example Query

Request

POST /query

Example:

{
 "query": "What does the constitution say about equality?"
}

Response:

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
🎯 Use Cases

Legal research assistants

Law student study tools

Civic education platforms

AI-powered legal search

Constitutional question answering

⚠️ Limitations

Currently supports Articles 1–35 only

Does not include case law or legal interpretation

Not a substitute for professional legal advice

🔮 Future Improvements

Add entire Constitution dataset

Case law retrieval

Better ranking of legal citations

Streaming responses

Authentication + deployment

Multi-document legal RAG

🧪 Built With

This project was prototyped using Antigravity for rapid AI-assisted development while implementing a custom architecture and idea.

📜 License

MIT License