# config.py — edit these to match your setup

# ── LM Studio ─────────────────────────────────────────────────────────────
# Make sure LM Studio is running and a model is loaded.
# Default server URL is http://localhost:1234
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"

# The model name shown in LM Studio (copy it exactly from the UI)
LM_STUDIO_MODEL = "deepseek/deepseek-r1-0528-qwen3-8b"

# ── Embeddings ─────────────────────────────────────────────────────────────
# Runs fully locally via sentence-transformers.
# Good default: small, fast, works well for English docs.
EMBED_MODEL = "all-MiniLM-L6-v2"

# ── Chroma vector store ────────────────────────────────────────────────────
# Where Chroma will persist its database on disk.
CHROMA_DIR = "./chroma_db"

# ── Files to index ─────────────────────────────────────────────────────────
# The folder (and subfolders) the agent will index.
WATCH_DIR = "/Users/moham/Desktop"

# Supported file extensions (add more as needed)
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".py", ".csv"}

# ── Retrieval ──────────────────────────────────────────────────────────────
# How many chunks to pull back per query
TOP_K = 20

# Chunk size and overlap for text splitting (in characters)
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 100

# ── Agent behaviour ────────────────────────────────────────────────────────
# Maximum retries before giving up on hallucination checks / query rewrites
MAX_RETRIES = 2
