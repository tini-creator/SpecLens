# SpecLens

A locally-hosted RAG pipeline for querying 3GPP telecommunications specifications in natural language. Point it at a spec PDF, build the index once, then ask questions like a senior engineer is reading the document.

No data leaves the local machine. No API keys required.

---

## How it works

SpecLens is built in three stages:

1. **Parse** — `document_parser.py` loads a 3GPP PDF and splits it into chunks using a telecom-aware strategy: it tries to break on section boundaries and paragraph breaks before ever splitting mid-sentence, preserving the integrity of Information Elements (IEs) and protocol descriptions.

2. **Embed** — `embedder.py` encodes each chunk with `sentence-transformers/all-MiniLM-L6-v2` and stores the resulting vectors in a local FAISS index.

3. **Query** — `rag_pipeline.py` takes a natural language question, retrieves the four most relevant chunks via FAISS, and passes them — alongside a strict grounding prompt — to `SmolLM2-1.7B-Instruct`. The model is instructed to answer only from the retrieved context, and to say so explicitly if the answer isn't there.

---

## Requirements

- Python 3.10+
- A 3GPP specification PDF (default: `data/raw/ts_38331.pdf`)
- CUDA-capable GPU recommended for indexing; CPU is fine for querying

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Quickstart

### 1. Place your PDF

```
data/raw/ts_38331.pdf
```

### 2. Build the FAISS index

```bash
python src/embedder.py --build-index
```

To use a different PDF:

```bash
python src/embedder.py --build-index --pdf-path data/raw/your_spec.pdf
```

This only needs to be run once. The index is saved to `vector_store/`.

### 3. Query the pipeline

**Single question:**

```bash
python src/main.py "What triggers a Radio Link Failure?"
```

**Interactive mode** (no question argument):

```bash
python src/main.py
```

```
Query> What is the purpose of the RRCReconfiguration message?

--- LLM ANSWER ---
The RRCReconfiguration message is used by the network to modify the UE's
radio resource configuration, including bearer setup, release, and modification...

--- RETRIEVED SOURCES ---
[1] 3GPP TS 38.331 - Page 312
[2] 3GPP TS 38.331 - Page 314
[3] 3GPP TS 38.331 - Page 189
[4] 3GPP TS 38.331 - Page 315
```

**GPU acceleration:**

```bash
python src/main.py --gpu "What are the T310 timer conditions?"
```

---

## Project structure

```
speclens/
├── data/
│   ├── raw/                  # Input PDFs
│   └── processed/            # Optional intermediate output
├── vector_store/             # Generated FAISS index (index.faiss, index.pkl)
├── src/
│   ├── __init__.py
│   ├── document_parser.py    # PDF loading and semantic chunking
│   ├── embedder.py           # Embedding model and FAISS index management
│   ├── rag_pipeline.py       # LangChain retrieval chain and LLM orchestration
│   └── main.py               # CLI entry point
├── requirements.txt
└── README.md
```

---

## Models

| Component | Model | Notes |
|---|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | 384-dim, runs on CPU |
| LLM | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 1.7B params, GPU recommended |

Both models are downloaded automatically from HuggingFace on first run.

---

## Design notes

**Why SmolLM2?** It's small enough to run on a laptop GPU (or slowly on CPU), follows instructions reliably, and stays within the retrieved context rather than hallucinating spec details.

**Why k=4 chunks?** 3GPP IEs often span multiple sections. Four chunks at 1500 characters each gives ~6000 characters of context — enough to cover a full IE definition and its surrounding protocol logic without overflowing the model's context window.

**Why local FAISS over a hosted vector DB?** Even though telecom specs are open-source, this example would be directly applicable to a scenario with proprietary/confidential technical documents. Keeping everything local means no query text or document content is transmitted externally.