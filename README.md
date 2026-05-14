# SpecLens: 3GPP Copilot

SpecLens is a locally hosted, Retrieval-Augmented Generation (RAG) pipeline designed to automate the analysis of dense telecommunications standards. By combining **LangChain**, local **FAISS** vector databases, and open-weight LLMs, this tool allows telecom engineers to query massive, unstructured 3GPP (4G/5G) specifications using natural language. This tool may be utilized for accelerating protocol stack diagnostics and feature extraction.



## Key Features



**Telecom-Aware Semantic Chunking:** Custom parsing logic designed specifically for 3GPP documents (e.g., TS 38.331). It intelligently splits text by protocol section headers and Information Elements (IEs) to preserve context, rather than relying on arbitrary character counts.

**100% Local Execution:** Utilizes local HuggingFace embeddings (`all-MiniLM-L6-v2`) and local LLM execution. No proprietary telecom data or queries are sent to external APIs (OpenAI, Anthropic, etc.), ensuring data privacy.

**High-Performance Retrieval:** Built on Meta's FAISS for rapid approximate nearest neighbor (ANN) search across thousands of embedded document chunks.



## Architecture



The pipeline is decoupled into distinct, scalable layers:

1\. **Ingestion (`document\_parser.py`):** Extracts raw text from 3GPP PDFs and applies telecom-specific recursive chunking.

2\. **Storage (`embedder.py`):** Converts chunks into dense vector representations and manages the local FAISS index.

3\. **Orchestration (`rag\_pipeline.py`):** Handles the LangChain retrieval chain, fusing the user's query with the retrieved context and injecting it into a strict system prompt.

4\. **Interface (`main.py`):** The CLI entry point for executing queries.



## Project Structure



```text

speclens/

├── data/

│   ├── raw/                  # Place the 3GPP TS 38.331 PDF here

│   └── processed/            # Intermediate chunked text (optional)

├── vector\_store/             # FAISS index artifacts (index.faiss, index.pkl)

├── src/                      

│   ├── \_\_init\_\_.py

│   ├── document\_parser.py    # Custom PDF parsing and semantic chunking

│   ├── embedder.py           # HuggingFace embedding logic

│   ├── rag\_pipeline.py       # LangChain QA chain orchestration

│   └── main.py               # CLI entry point

├── .env                      # Environment configurations (model paths)

├── requirements.txt          # Python dependencies

└── README.md                 # Project documentation

