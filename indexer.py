# indexer_v2.py — priority ordering and rich metadata

import json
import os
import hashlib
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import SUPPORTED_EXTENSIONS, CHUNK_SIZE, CHUNK_OVERLAP, CHROMA_DIR

_HASH_STORE = Path(CHROMA_DIR) / "indexed_hashes.json"

# Index these files before others in the same directory
_PRIORITY_STEMS = {"readme", "index", "__init__", "main", "app"}


# ── Hash persistence ────────────────────────────────────────────────────────

def _load_hashes() -> set[str]:
    if _HASH_STORE.exists():
        return set(json.loads(_HASH_STORE.read_text()))
    return set()


def _save_hashes(hashes: set[str]) -> None:
    _HASH_STORE.parent.mkdir(parents=True, exist_ok=True)
    _HASH_STORE.write_text(json.dumps(list(hashes)))


# ── Hashing ──────────────────────────────────────────────────────────────────

def _file_hash(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _priority_key(fname: str) -> int:
    return 0 if Path(fname).stem.lower() in _PRIORITY_STEMS else 1


# ── File loading ─────────────────────────────────────────────────────────────

def _load_file(path: str) -> str:
    suffix = Path(path).suffix.lower()

    if suffix == ".pdf":
        try:
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(path)
            pages = loader.load()
            return "\n\n".join(p.page_content for p in pages)
        except Exception as e:
            print(f"    ⚠️  Could not read PDF {path}: {e}")
            return ""

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        print(f"    ⚠️  Could not read {path}: {e}")
        return ""


# ── Directory traversal ──────────────────────────────────────────────────────

_SKIP_DIRS = {
    # Python
    ".venv", "venv", "env", "__pycache__", "site-packages",
    ".pytest_cache", ".tox", "htmlcov", "eggs",
    # JS / web
    "node_modules", "dist", "build", ".next", ".nuxt", ".svelte-kit",
    # Version control & editors
    ".git", ".idea", ".vscode",
    # Build outputs
    "target", "out", "bin", "obj",
    # macOS junk
    "__MACOSX",
    # Dependency managers
    "vendor", "Pods",
    # Xcode
    "DerivedData",
    # Misc
    "tmp", "temp", "logs", "log", "cache", ".cache",
    "chroma_db", "wordpress",
}


def _collect_files(directory: str) -> List[str]:
    """Recursively collect supported files, priority files first within each folder."""
    found: List[str] = []
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS and not d.startswith(".")]
        supported = [f for f in files if Path(f).suffix.lower() in SUPPORTED_EXTENSIONS]
        found.extend(os.path.join(root, f) for f in sorted(supported, key=_priority_key))
    return found


# ── Main entry point ──────────────────────────────────────────────────────────

def index_files(vectorstore: Chroma, watch_dir: str) -> None:
    """
    Walk watch_dir and embed new/changed files into the vector store.
    - Skips files whose hash is already stored (persisted across restarts)
    - Indexes folder manifests first (coarse-grained, for two-level retrieval)
    - Indexes priority files (README, index, __init__) before others per folder
    - Attaches rich metadata: folder, filename, extension, depth, type
    """
    if not os.path.exists(watch_dir):
        print(f"    ⚠️  Watch directory '{watch_dir}' does not exist — creating it.")
        os.makedirs(watch_dir)
        return

    indexed_hashes = _load_hashes()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    files = _collect_files(watch_dir)
    if not files:
        print(f"    ℹ️  No supported files found in '{watch_dir}'.")
        return

    new_count = 0
    for fpath in files:
        fhash = _file_hash(fpath)
        if fhash in indexed_hashes:
            continue

        text = _load_file(fpath)
        if not text.strip():
            continue

        p = Path(fpath)
        try:
            depth = len(p.relative_to(watch_dir).parts) - 1
        except ValueError:
            depth = 0

        chunks = splitter.create_documents(
            texts=[text],
            metadatas=[{
                "source":    fpath,
                "hash":      fhash,
                "folder":    str(p.parent),
                "filename":  p.name,
                "extension": p.suffix.lower(),
                "depth":     depth,
                "type":      "file",
            }],
        )

        ids = [f"{fhash}_{i}" for i in range(len(chunks))]
        batch_size = 5000
        for i in range(0, len(chunks), batch_size):
            vectorstore.add_documents(chunks[i:i + batch_size], ids=ids[i:i + batch_size])

        indexed_hashes.add(fhash)
        new_count += 1
        print(f"    ✅  Indexed: {fpath} ({len(chunks)} chunks)")

    _save_hashes(indexed_hashes)

    if new_count == 0:
        print("    ✅  All files already up to date.")
    else:
        print(f"    📥  {new_count} file(s) newly indexed.")
