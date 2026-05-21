# Autonomous Legal RAG Agent & Analytics Engine

An advanced Agentic RAG (Retrieval-Augmented Generation) system for constitutional and legal document reasoning using ReAct orchestration, dual-vector retrieval pipelines, ChromaDB, and Groq Llama-3.

The system supports:
- Semantic Retrieval
- Parent-Child Context Retrieval
- Dynamic PDF Upload Analysis
- Temporary In-Memory Vector Databases
- ReAct-based Agentic Reasoning
- Evaluation Metrics for RAG Performance

---

# Features

## Agentic ReAct Workflow
Implements iterative:
- Thought
- Action
- Observation
- Final Answer

reasoning loops for autonomous legal analysis.

## Dual Retrieval Architecture
Two specialized retrieval pipelines:

### Semantic Retrieval
Retrieves highly precise sentence-level legal fragments.

### Parent-Child Retrieval
Retrieves broader contextual constitutional/legal sections for structural reasoning.

## Dynamic PDF Upload Pipeline
Users can upload PDFs and query documents in real-time using:
- Ephemeral ChromaDB
- Temporary in-memory vector stores
- Runtime indexing

without polluting the permanent database.

## Evaluation Framework
Tracks:
- Accuracy
- Faithfulness
- Relevance
- Failure Cases

for RAG quality analysis.

---

# Tech Stack

- Python
- Streamlit
- Groq API
- Gemini API
- Llama-3
- ChromaDB
- Sentence Transformers
- LangChain
- PyPDF
- ReAct Agents

---

# Project Architecture

```text
User Query
      ↓
Agent Orchestrator (ReAct Loop)
     ↓
Tool Selection
 ┌───────────────┬────────────────┐
 │               │                │
Semantic Vault   Parent-Child Vault
 │               │
Precise Search   Contextual Search
 └───────────────┴────────────────┘
     ↓
Observation Injection
     ↓
Final Legal Synthesis
```

---

# Folder Structure

```text
project/
│
├── app.py
├── agent_orchestrator.py
├── rag_engine.py
├── rag_engine_v2.py
├── upload_pipeline.py
├── evaluation.ipynb
├── chroma_storage/
├── chroma_storage_v2/
├── requirements.txt
└── README.md
```

---

# Installation

## Clone Repository

git clone https://github.com/pranay0026/LegalRag.git
cd LegalRag

## Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

#### Windows
```bash
venv\\Scripts\\activate
```

#### Linux/Mac
```bash
source venv/bin/activate
```

---

# Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_key_here
GROQ_API_KEY3=your_key_here
```

---

# Run Application

```bash
streamlit run app.py
```

---

# Evaluation Metrics

Current benchmark results:

| Metric | Score |
|---|---|
| Accuracy | 4.00 / 5 |
| Faithfulness | 3.95 / 5 |
| Relevance | 4.00 / 5 |

---

# Example Queries

## Constitution Database Queries

- Powers of CMs under Article 162
- Fundamental Rights under Article 21
- Governor discretionary powers
- Emergency provisions in Constitution

## Dynamic PDF Queries

- Summarize this agreement
- Identify legal risks
- Extract obligations from contract
- Explain termination clauses

---

# Future Improvements

- Hybrid Search (BM25 + Vector Search)
- Reranking Pipelines
- Streaming Responses
- Conversational Memory
- FastAPI Backend
- Multi-document Retrieval

---

**#Images**
![alt text](<images/Screenshot 2026-05-21 193854.png>)


# Author

Pranay Koppineedi

---

# License

This project is intended for educational and research purposes.
