import re
import streamlit as st

def render_sources_block(text: str):
    """Render the 'Sources:' section if present in a final answer."""
    if not text:
        return
    # naive parse: split on 'Sources:'
    parts = text.split('\nSources:')
    if len(parts) < 2:
        return
    sources = parts[1].strip().splitlines()
    with st.expander('Sources'):
        for line in sources:
            st.markdown(f"- {line.lstrip('- ').strip()}")

def write_chat_history(history):
    for role, content in history:
        with st.chat_message('user' if role=='user' else 'assistant'):
            st.markdown(content)
