from sqlalchemy import create_engine, text as sql_text

# --- Helper: read stored vector dimension from pgvector collection
def fetch_collection_dim(pg_url: str, collection_name: str) -> int | None:
    """Return the stored vector dimension for the collection, if any."""
    try:
        engine = create_engine(pg_url)
        with engine.begin() as conn:
            dims = conn.execute(sql_text(
                """
                SELECT vector_dims(e.embedding) AS dims
                FROM langchain_pg_embedding e
                JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                WHERE c.name = :name
                LIMIT 1
                """
            ), {"name": collection_name}).scalar()
        return int(dims) if dims is not None else None
    except Exception:
        return None
    
    
def _extract_sources_block(text: str) -> list[str]:
    if not text:
        return []
    # Expect replanner to append a "Sources:" section for RAG answers.
    parts = text.split("\nSources:")
    if len(parts) < 2:
        return []
    lines = parts[1].strip().splitlines()
    return [l.lstrip("- ").strip() for l in lines if l.strip()]

