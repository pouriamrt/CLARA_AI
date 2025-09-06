import asyncio
import uuid
import streamlit as st

from core.config import AppConfig
from core.graph import build_app
from components.ui import render_sources_block

st.set_page_config(page_title="BHI Planner – Streamlit", page_icon="🧠", layout="wide")

st.title("🧠 Biomedical Assistant (Planner + Tools)")
st.caption("RAG over pgvector + SQL tools, orchestrated with LangGraph.")

# Sidebar config
with st.sidebar:
    st.header("Configuration")
    cfg = AppConfig.from_env()
    st.text_input("OpenAI API Key", value=cfg.openai_api_key, type="password", key="OPENAI_API_KEY")
    st.text_input("SQL Database URL", value=cfg.sql_database_url, key="SQL_DATABASE_URL")
    st.text_input("PGVector URL", value=cfg.pgvector_url, key="PGVECTOR_URL")
    st.text_input("PGVector Collection", value=cfg.pgvector_collection, key="PGVECTOR_COLLECTION")
    st.text_input("Chat Model", value=cfg.openai_chat_model, key="OPENAI_CHAT_MODEL")
    st.text_input("Embedding Model", value=cfg.openai_embed_model, key="OPENAI_EMBED_MODEL")
    st.text_input("Top K", value=cfg.top_k, key="TOP_K")
    st.divider()
    if st.button("Rebuild Graph"):
        st.session_state["_graph_built"] = False

def current_config() -> AppConfig:
    return AppConfig(
        openai_api_key=st.session_state.get("OPENAI_API_KEY",""),
        sql_database_url=st.session_state.get("SQL_DATABASE_URL",""),
        pgvector_url=st.session_state.get("PGVECTOR_URL",""),
        pgvector_collection=st.session_state.get("PGVECTOR_COLLECTION","state_of_union_vectors"),
        openai_chat_model=st.session_state.get("OPENAI_CHAT_MODEL","gpt-4.1-mini"),
        openai_embed_model=st.session_state.get("OPENAI_EMBED_MODEL","text-embedding-3-large"),
        top_k=st.session_state.get("TOP_K",10),
    )

# Build graph once per session or after 'Rebuild'
if "_graph_built" not in st.session_state or not st.session_state["_graph_built"]:
    st.session_state["cfg"] = current_config()
    st.session_state["graph"] = build_app(st.session_state["cfg"])
    st.session_state["_graph_built"] = True
    st.session_state.setdefault("thread_id", str(uuid.uuid4()))
    st.session_state.setdefault("chat", [])  # [(role, content), ...]

# Render previous chat
for role, content in st.session_state["chat"]:
    with st.chat_message(role):
        st.markdown(content)

# Chat input
if prompt := st.chat_input("Ask something… (the agent will plan + use tools)"):
    st.session_state["chat"].append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run the graph
    app = st.session_state["graph"]
    cfg = st.session_state["cfg"]
    config = {"configurable": {"thread_id": st.session_state["thread_id"]}, "recursion_limit": 50}

    # Call invoke (single-shot). Could switch to streaming if desired.
    result = asyncio.run(app.ainvoke({"input": prompt, "plan": [], "history": []}, config=config))
    final = result.get("response") or "No response."
    with st.chat_message("assistant"):
        st.markdown(final)
        render_sources_block(final)

    st.session_state["chat"].append(("assistant", final))
