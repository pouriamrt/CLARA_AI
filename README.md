# BHI Planner – Streamlit

This app serves the notebook logic as a Streamlit chatbot using LangGraph:
- **Planner → Step Executor → Replanner** loop
- **Tools**: pgvector-based RAG retriever + SQL toolkit
- **Models**: OpenAI chat + embeddings

## Run locally

```bash
cd streamlit_bhi_app
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # then fill in your keys/URLs
streamlit run streamlit_app.py
```

## Configuration

Use `.env` or `.streamlit/secrets.toml`:
- `OPENAI_API_KEY`
- `SQL_DATABASE_URL` (optional)
- `PGVECTOR_URL`
- `PGVECTOR_COLLECTION` (e.g., `state_of_union_vectors`)
- `OPENAI_CHAT_MODEL` (default: `gpt-4.1-mini`)
- `OPENAI_EMBED_MODEL` (default: `text-embedding-3-large`)

## Notes

- The retriever tool always returns a string with a `CONTEXT:` block and a `SOURCES_JSON=[...]` tail. The replanner converts this into a friendly **Sources** section in the final answer.
- The SQL toolkit follows a strict schema-first policy via prompt instructions.
