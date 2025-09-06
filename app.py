import uuid
import json
import chainlit as cl

from core.config import AppConfig
from core.graph import build_app
from core.retriever import fetch_collection_dim
from langchain_openai import OpenAIEmbeddings

from core.helper import _extract_sources_block, SourceCatcher, _fmt_source


SESSION_GRAPH = "graph"
SESSION_CFG = "cfg"
SESSION_THREAD = "thread_id"

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
    usr = cl.user_session.get("user")
    
    await cl.Message(
        content=(
            f"Hello, {usr.identifier.split('@')[0]}! 👋"
        )
    ).send()

@cl.on_message
async def on_message(message: cl.Message):
    text = message.content.strip()

    graph = cl.user_session.get(SESSION_GRAPH)
    thread_id = cl.user_session.get(SESSION_THREAD)
    cfg: AppConfig = cl.user_session.get(SESSION_CFG)

    thinking = cl.Message(content="Thinking…")
    await thinking.send()
    
    langchain_cb = cl.LangchainCallbackHandler(
        stream_final_answer=True,
        answer_prefix_tokens=["**Answer:**\n\n"],
    )

    catcher = SourceCatcher()
    
    # Single-shot invoke (you could switch to streaming with astream_events to show tool steps)
    result = await graph.ainvoke(
        {"input": text, "plan": [], "history": []},
        config={"configurable": {"thread_id": thread_id}, "recursion_limit": 50, "callbacks": [langchain_cb, catcher]},
    )
    final = result.get("response") or "No response."

    sources = catcher.sources or _extract_sources_block(final)
    # Render final; optionally split to hide sources block and show elements separately
    if sources:
        # Show answer without the Sources block duplicated
        answer_only = final.split("\nSources:")[0].rstrip()
        thinking.content = answer_only
        await thinking.update()
        await cl.Message(
            content="**Sources**",
            elements=[cl.Text(name=f"Source {i+1}", content=_fmt_source(s)) for i, s in enumerate(sources)],
        ).send()
    else:
        thinking.content = final
        await thinking.update()
