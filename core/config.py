from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv(override=True)

def _get(name: str, default: Optional[str] = None) -> str:
    return os.environ.get(name, default) if default is not None else os.environ.get(name, "")

@dataclass(frozen=True)
class AppConfig:
    openai_api_key: str
    sql_database_url: str
    pgvector_url: str
    pgvector_collection: str
    openai_chat_model: str = "gpt-4.1-mini"
    openai_embed_model: str = "text-embedding-3-small"
    top_k: int = 10

    @staticmethod
    def from_env() -> "AppConfig":
        return AppConfig(
            openai_api_key=_get("OPENAI_API_KEY", ""),
            sql_database_url=_get("SQL_DATABASE_URL", ""),
            pgvector_url=_get("PGVECTOR_URL", ""),
            pgvector_collection=_get("PGVECTOR_COLLECTION", "state_of_union_vectors"),
            openai_chat_model=_get("OPENAI_CHAT_MODEL", "gpt-4.1-mini"),
            openai_embed_model=_get("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
            top_k=_get("TOP_K", 10),
        )
