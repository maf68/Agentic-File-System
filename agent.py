"""
Local RAG Agent using LangGraph
- LLM: LM Studio (local OpenAI-compatible server)
- Embeddings: sentence-transformers (local)
- Vector store: Chroma (local, persistent)
"""

import os
from typing import Annotated, List, Literal
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.documents import Document

from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages

from indexer import index_files
from config import (
    LM_STUDIO_BASE_URL,
    LM_STUDIO_MODEL,
    EMBED_MODEL,
    CHROMA_DIR,
    WATCH_DIR,
    TOP_K,
    MAX_RETRIES,
)

# ── LLM (LM Studio exposes an OpenAI-compatible endpoint) ──────────────────
llm = ChatOpenAI(
    base_url=LM_STUDIO_BASE_URL,
    api_key="lm-studio",           # LM Studio ignores the key, but the param is required
    model=LM_STUDIO_MODEL,
    temperature=0,
)

# ── Embeddings (fully local, no network call) ──────────────────────────────
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

# ── Vector store ───────────────────────────────────────────────────────────
vectorstore = Chroma(
    collection_name="local_files",
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})


# ── State ──────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages:        Annotated[List[BaseMessage], add_messages]
    question:        str
    retrieved_docs:  List[Document]
    generation:      str
    route:           Literal["rag", "direct"]
    grounded:        bool
    retry_count:     int


# ── Node helpers ───────────────────────────────────────────────────────────

def _format_docs(docs: List[Document]) -> str:
    return "\n\n---\n\n".join(
        f"[{doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    )

def _call_llm(system: str, user: str, history: List[BaseMessage] | None = None) -> str:
    msgs = [SystemMessage(content=system)]
    if history:
        msgs += history
    msgs.append(HumanMessage(content=user))
    response = llm.invoke(msgs)
    return response.content.strip()


# ── Nodes ──────────────────────────────────────────────────────────────────

def file_indexer_node(state: AgentState) -> AgentState:
    """Scan WATCH_DIR, embed new/changed files into Chroma."""
    print("📂  Indexing local files...")
    index_files(vectorstore, WATCH_DIR)
    return state


def router_node(state: AgentState) -> AgentState:
    """LLM decides: does this question need file retrieval?"""
    print("🔀  Routing question...")
    system = (
        "You are a router. Given a user question, decide whether answering it "
        "requires searching local files and documents, or whether you can answer "
        "directly from general knowledge.\n"
        "Reply with ONLY one word: 'rag' or 'direct'."
    )
    answer = _call_llm(system, state["question"]).lower()
    route = "rag" if "rag" in answer else "direct"
    print(f"    → route: {route}")
    return {**state, "route": route}


def retrieval_node(state: AgentState) -> AgentState:
    """Embed the question and pull top-k chunks from Chroma."""
    print("🔍  Retrieving chunks...")
    docs = retriever.invoke(state["question"])
    print(f"    → {len(docs)} chunks retrieved")
    return {**state, "retrieved_docs": docs}


def rag_reflection_node(state: AgentState) -> AgentState:
    """Score each chunk; keep only relevant ones. Rewrite query if none pass."""
    print("📋  Grading chunks...")
    question = state["question"]
    kept = []
    for doc in state["retrieved_docs"]:
        system = (
            "You are a relevance grader. Given a question and a document chunk, "
            "decide if the chunk is useful for answering the question.\n"
            "Reply with ONLY 'yes' or 'no'."
        )
        user = f"Question: {question}\n\nChunk:\n{doc.page_content}"
        verdict = _call_llm(system, user).lower()
        if "yes" in verdict:
            kept.append(doc)

    print(f"    → {len(kept)}/{len(state['retrieved_docs'])} chunks kept")

    if not kept and state["retry_count"] < MAX_RETRIES:
        # Rewrite the query and try retrieval again
        print("    → No good chunks — rewriting query for retry...")
        rewritten = _call_llm(
            "Rewrite the following question to improve document retrieval. "
            "Return only the rewritten question, nothing else.",
            question,
        )
        new_docs = retriever.invoke(rewritten)
        return {
            **state,
            "question": rewritten,
            "retrieved_docs": new_docs,
            "retry_count": state["retry_count"] + 1,
        }

    return {**state, "retrieved_docs": kept}


def generate_node(state: AgentState) -> AgentState:
    """LLM synthesises an answer from the surviving chunks."""
    print("✍️   Generating answer...")
    context = _format_docs(state["retrieved_docs"])
    system = (
        "You are a helpful assistant. Answer the question using ONLY the provided "
        "document excerpts. If the answer is not in the documents, say so.\n\n"
        f"Documents:\n{context}"
    )
    answer = _call_llm(system, state["question"], history=state["messages"])
    return {**state, "generation": answer, "messages": [AIMessage(content=answer)]}


def direct_answer_node(state: AgentState) -> AgentState:
    """LLM answers from its own knowledge, no retrieval."""
    print("💬  Answering directly...")
    answer = _call_llm(
        "You are a helpful assistant. Answer the question concisely.",
        state["question"],
        history=state["messages"],
    )
    return {**state, "generation": answer, "messages": [AIMessage(content=answer)]}


def hallucination_check_node(state: AgentState) -> AgentState:
    """Check whether the generated answer is grounded in the retrieved chunks."""
    print("🔎  Checking for hallucinations...")
    context = _format_docs(state["retrieved_docs"])
    system = (
        "You are a grounding checker. Given document excerpts and an answer, "
        "decide if every factual claim in the answer is supported by the documents.\n"
        "Reply with ONLY 'grounded' or 'hallucinated'."
    )
    user = (
        f"Documents:\n{context}\n\n"
        f"Answer:\n{state['generation']}"
    )
    verdict = _call_llm(system, user).lower()
    grounded = "grounded" in verdict
    print(f"    → {'✅ grounded' if grounded else '❌ hallucinated'}")
    return {**state, "grounded": grounded}


# ── Conditional edge functions ─────────────────────────────────────────────

def route_after_router(state: AgentState) -> Literal["file_indexer_node", "direct_answer_node"]:
    return "file_indexer_node" if state["route"] == "rag" else "direct_answer_node"


def route_after_grading(state: AgentState) -> Literal["generate_node", "rag_reflection_node"]:
    """If chunks look good, generate. Otherwise loop back to grade again after rewrite."""
    if state["retrieved_docs"]:
        return "generate_node"
    return "rag_reflection_node"   # will rewrite inside rag_reflection_node


def route_after_hallucination_check(state: AgentState) -> Literal["generate_node", "__end__"]:
    if state["grounded"] or state["retry_count"] >= MAX_RETRIES:
        return "__end__"
    return "generate_node"


# ── Build graph ────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    g = StateGraph(AgentState)

    g.add_node("file_indexer_node",       file_indexer_node)
    g.add_node("router_node",             router_node)
    g.add_node("retrieval_node",          retrieval_node)
    g.add_node("rag_reflection_node",         rag_reflection_node)
    g.add_node("generate_node",           generate_node)
    g.add_node("direct_answer_node",      direct_answer_node)
    # g.add_node("hallucination_check_node", hallucination_check_node)

    g.add_edge(START,                  "router_node")

    g.add_conditional_edges("router_node", route_after_router)

    g.add_edge("file_indexer_node",    "retrieval_node")
    g.add_edge("retrieval_node",       "rag_reflection_node")
    g.add_conditional_edges("rag_reflection_node", route_after_grading)
    g.add_edge("generate_node",        END)
    # g.add_conditional_edges("hallucination_check_node", route_after_hallucination_check)

    g.add_edge("direct_answer_node",   END)

    return g.compile()


# ── Entry point ────────────────────────────────────────────────────────────

def ask(question: str, history: List[BaseMessage] | None = None) -> tuple[str, List[BaseMessage]]:
    graph = build_graph()
    prior = history or []
    initial_state: AgentState = {
        "messages":       prior + [HumanMessage(content=question)],
        "question":       question,
        "retrieved_docs": [],
        "generation":     "",
        "route":          "rag",
        "grounded":       False,
        "retry_count":    0,
    }
    final_state = graph.invoke(initial_state)
    return final_state["generation"], final_state["messages"]


if __name__ == "__main__":
    messages: List[BaseMessage] = []
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break
        answer, messages = ask(q, messages)
        print(f"\nAgent: {answer}\n")
