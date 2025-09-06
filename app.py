import uuid
import json
import chainlit as cl

from core.config import AppConfig
from core.graph import build_app
from core.retriever import fetch_collection_dim
from langchain_openai import OpenAIEmbeddings

SESSION_GRAPH = "graph"
SESSION_CFG = "cfg"
SESSION_THREAD = "thread_id"

def _extract_sources_block(text: str) -> list[str]:
    if not text:
        return []
    # Expect replanner to append a "Sources:" section for RAG answers.
    parts = text.split("\nSources:")
    if len(parts) < 2:
        return []
    lines = parts[1].strip().splitlines()
    return [l.lstrip("- ").strip() for l in lines if l.strip()]

@cl.oauth_callback
def oauth_callback(provider_id, token, raw_user_data, default_user):
    return default_user

@cl.password_auth_callback
def auth_callback(username, password):
    return cl.User(identifier="admin", metadata={"role": "admin"}) if (username, password) == ("admin", "admin") else None

@cl.on_chat_start
async def on_chat_start():
    cfg = AppConfig.from_env()
    cl.user_session.set(SESSION_CFG, cfg)
    cl.user_session.set(SESSION_THREAD, str(uuid.uuid4()))
    cl.user_session.set(SESSION_GRAPH, build_app(cfg))

    await cl.Message(
        content=(
            "🧠 **Biomedical Assistant (Chainlit)** is ready.\n"
            "- RAG over pgvector + SQL tools\n"
            "- Planner → Step executor → Replanner\n\n"
            "Type your question, or `/validate` to check pgvector dimensions."
        )
    ).send()

@cl.on_message
async def on_message(message: cl.Message):
    text = message.content.strip()

    # Slash command: /validate to check dim mismatch
    if text.lower().startswith("/validate"):
        cfg: AppConfig = cl.user_session.get(SESSION_CFG)
        qdim, sdim, err = None, None, None
        try:
            emb = OpenAIEmbeddings(api_key=cfg.openai_api_key)#, model=cfg.openai_embed_model)
            qdim = len(emb.embed_query("ping"))
        except Exception as e:
            err = f"Embeddings init error: {e}"

        try:
            sdim = fetch_collection_dim(cfg.pgvector_url, cfg.pgvector_collection)
        except Exception as e:
            err = (err or "") + f" ; fetch_collection_dim error: {e}"

        advice = []
        if sdim and qdim and sdim != qdim:
            advice.append(
                "- Dimension mismatch. Options:\n"
                "  1) Switch OPENAI_EMBED_MODEL to the one used for this collection and re-run.\n"
                "  2) Create a NEW collection and re-embed with the current model.\n"
                "  3) If using TE3-* models, set `dimensions=` to match stored size during (re)indexing."
            )
        elif sdim:
            advice.append("- Dimensions look compatible ✅")
        else:
            advice.append("- Stored dimension unknown (empty collection or insufficient permissions).")

        await cl.Message(
            content=(
                f"**Validation**\n"
                f"- Query dim: `{qdim or 'unknown'}`\n"
                f"- Stored dim: `{sdim or 'unknown'}`\n"
                f"{('Errors: ' + err) if err else ''}\n\n" + "\n".join(advice)
            )
        ).send()
        return

    # Normal question flow
    graph = cl.user_session.get(SESSION_GRAPH)
    thread_id = cl.user_session.get(SESSION_THREAD)
    cfg: AppConfig = cl.user_session.get(SESSION_CFG)

    thinking = cl.Message(content="Thinking…")
    await thinking.send()
    
    langchain_cb = cl.LangchainCallbackHandler(
        stream_final_answer=True,
        answer_prefix_tokens=["**Answer:**\n\n"],
    )

    # Single-shot invoke (you could switch to streaming with astream_events to show tool steps)
    result = await graph.ainvoke(
        {"input": text, "plan": [], "history": []},
        config={"configurable": {"thread_id": thread_id}, "recursion_limit": 50, "callbacks": [langchain_cb]},
    )
    final = result.get("response") or "No response."

    sources = _extract_sources_block(final)
    # Render final; optionally split to hide sources block and show elements separately
    if sources:
        # Show answer without the Sources block duplicated
        answer_only = final.split("\nSources:")[0].rstrip()
        thinking.content = answer_only
        await thinking.update()
        await cl.Message(
            content="**Sources**",
            elements=[cl.Text(name=f"Source {i+1}", content=s) for i, s in enumerate(sources)],
        ).send()
    else:
        thinking.content = final
        await thinking.update()
