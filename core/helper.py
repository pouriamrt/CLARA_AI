from sqlalchemy import create_engine, text as sql_text
import json, ast
from typing import Any, Dict, List
from langchain.callbacks.base import BaseCallbackHandler

def _largest_json_block(s: str) -> str:
    l, r = s.find("{"), s.rfind("}")
    return s[l:r+1] if (l != -1 and r != -1 and r > l) else s

def _content_to_text(parts: Any) -> str | None:
    """
    LangChain tool outputs may come as:
      - str
      - list[str]
      - list[{'type':'text','text':'...'}]
    Consolidate to a single string if possible.
    """
    if isinstance(parts, str):
        return parts
    if isinstance(parts, list):
        buf: List[str] = []
        for p in parts:
            if isinstance(p, str):
                buf.append(p)
            elif isinstance(p, dict) and ("text" in p or p.get("type") == "text"):
                buf.append(str(p.get("text", "")))
        return "\n".join([b for b in buf if b]) if buf else None
    return None

def _coerce_to_dict(output: Any) -> Dict | None:
    # 1) Already a dict?
    if isinstance(output, dict):
        return output

    # 2) Message-like object with `.content`?
    content = None
    if hasattr(output, "content"):
        content = _content_to_text(getattr(output, "content"))

    # 3) Or just a plain string?
    if content is None and isinstance(output, str):
        content = output

    if not content:
        return None

    s = content.strip()

    # 4) Try strict JSON
    try:
        return json.loads(s)
    except Exception:
        pass

    # 5) Try largest {...} block
    block = _largest_json_block(s)
    try:
        return json.loads(block)
    except Exception:
        pass

    # 6) Try Python literal (handles single quotes)
    try:
        lit = ast.literal_eval(block)
        return lit if isinstance(lit, dict) else None
    except Exception:
        return None


class SourceCatcher(BaseCallbackHandler):
    def __init__(self, target_tool_name: str = "retrieve_paper_chunks"):
        self.sources: list = []
        self.docs: list = []
        self.payload: dict | None = None
        self._last_tool_name = None
        self._target = target_tool_name

    # Keep async hooks since the app is async
    async def on_tool_start(self, serialized, input_str, **kwargs):
        name = (serialized or {}).get("name") or kwargs.get("name")
        self._last_tool_name = name
        print(f"[SourceCatcher] Tool started: {name}")

    async def on_tool_end(self, output, **kwargs):
        if self._last_tool_name != self._target:
            return
        data = _coerce_to_dict(output)

        if isinstance(data, dict):
            self.payload = data
            if isinstance(data.get("sources"), list):
                self.sources = data["sources"]
            if isinstance(data.get("docs"), list):
                self.docs = data["docs"]
                

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


def _fmt_source(s) -> str:
    # Handles dicts or preformatted strings
    if isinstance(s, dict):
        title  = s.get("title") or s.get("Title") or "Untitled"
        auth   = s.get("authors") or s.get("Author") or "Unknown"
        year   = s.get("publication_year") or s.get("Publication Year") or "n.d."
        page   = s.get("page")
        page_s = f", p.{page}" if page is not None else ""
        return f"{title} — {auth} ({year}){page_s}"
    return str(s)

