# Local RAG Agent

RAG agent for local folders and files built with LangGraph.
- **LLM**: LM Studio (any model you have loaded)
- **Embeddings**: `all-MiniLM-L6-v2` via sentence-transformers — runs on CPU, no GPU needed
- **Vector store**: Chroma, persisted to disk

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure LM Studio

1. Open LM Studio and load any model (Mistral, LLaMA, Phi, etc.)
2. Go to **Local Server** (the `<->` icon on the left sidebar)
3. Click **Start Server** — it starts on `http://localhost:1234` by default
4. Copy the model name exactly as shown and paste it into `config.py` → `LM_STUDIO_MODEL`

### 3. Point to your files

Edit `config.py`:

```python
WATCH_DIR = "./my_files"   # change this to your actual folder path
```

Supported file types by default: `.txt`, `.md`, `.pdf`, `.py`, `.csv`
Add more extensions in `config.py` → `SUPPORTED_EXTENSIONS`.

### 4. Run

```bash
python agent.py "What did I write about the project deadline?"
```

Or import and call from your own script:

```python
from agent import ask

answer = ask("Summarise my notes on the budget meeting.")
print(answer)
```

---

## How it works

```
START
  │
  ▼
File indexer   → scans WATCH_DIR, embeds new/changed files into Chroma
  │
  ▼
Router         → LLM decides: needs file retrieval, or can answer directly?
  │
  ├─ needs docs ──► Retrieval → Grade docs ──► Generate → Hallucination check
  │                                 │ (retry if no good chunks)      │
  │                                 └─────────────────────────────── ┘
  │                                                                   │
  └─ can answer ──► Direct answer                                     │
                         │                                            │
                         └──────────────── END ◄─────────────────────┘
```

---

## Tips

- **Speed**: The router, grader, and hallucination checker use small yes/no prompts.
  Loading a faster model in LM Studio for these calls helps a lot.
- **Quality**: Larger context window models work better for the generate node.
- **Files not being found**: Check that `WATCH_DIR` path is correct and files have
  supported extensions.
- **Chroma persists**: The `chroma_db/` folder is written to disk. Delete it to
  force a full re-index.
# Agentic-File-System
