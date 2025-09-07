from __future__ import annotations
from typing import List, Dict, Any
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_community.vectorstores import PGVector
# from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.retrievers import SelfQueryRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (
    LLMChainExtractor,
    EmbeddingsFilter,
    DocumentCompressorPipeline,
)
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from core.helper import fetch_collection_dim

class RetrieveInput(BaseModel):
    query: str = Field(..., description="The user query to retrieve relevant paper chunks for.")

# Metadata schema used by SelfQueryRetriever
def build_metadata_info() -> List[AttributeInfo]:
    return [
        AttributeInfo(
            name="Publication Year",
            description="The year that the paper was published.",
            type="integer",
        ),
        AttributeInfo(
            name="Date Added",
            description="The year that the paper was added to the collection.",
            type="integer",
        ),
        AttributeInfo(
            name="Author",
            description="Authors of the paper, it could be couple of people.",
            type="string",
        ),
        AttributeInfo(
            name="Title", 
            description="Title of the paper that the paper is about.", 
            type="string",
        ),
        AttributeInfo(
            name="Cleaned_Abs", 
            description="Abstract of the paper that the paper is about.", 
            type="string",
        ),
        AttributeInfo(
            name="Population", 
            description="Whether the Population is mentioned in the paper. P flag in PICOS is the string True or False.", 
            type="string",
        ),
        AttributeInfo(
            name="Intervention", 
            description="Whether the Intervention is mentioned in the paper. I flag in PICOS is the string True or False.", 
            type="string",
        ),
        AttributeInfo(
            name="Comparator", 
            description="Whether the Comparator is mentioned in the paper. C flag in PICOS is the string True or False.", 
            type="string",
        ),
        AttributeInfo(
            name="Outcome", 
            description="Whether the Outcome is mentioned in the paper. O flag in PICOS is the string True or False.", 
            type="string",
        ),
        AttributeInfo(
            name="Study Design", 
            description="Whether the Study Design is mentioned in the paper. S flag in PICOS is the string True or False.", 
            type="string",
        ),
        AttributeInfo(
            name="Qualification", 
            description="Whether the paper is PICOS compliant or not. It is the string Qualified or Not Qualified.", 
            type="string",
        ),
    ]

def build_retriever_tool(
    pgvector_url: str,
    collection_name: str,
    openai_api_key: str,
    openai_chat_model: str,
    openai_embed_model: str,
    top_k: int = 10,
) -> StructuredTool:
    # Vector store
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)#, model=openai_embed_model)
    vectorstore = PGVector(
        embedding_function=embeddings,
        collection_name=collection_name,
        connection_string=pgvector_url,
        use_jsonb=True,
    )

    llm = ChatOpenAI(model=openai_chat_model, temperature=0, api_key=openai_api_key)

    # 1) Self-query over metadata + text
    base_retriever = SelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=vectorstore,
        document_contents="medical research papers",
        metadata_field_info=build_metadata_info(),
        search_kwargs={"k": top_k, "fetch_k": max(int(top_k)*4, 25), "mmr": True, "lambda_mult": 0.5},
    )

    extractor = LLMChainExtractor.from_llm(llm)
    emb_filter = EmbeddingsFilter(
        embeddings=embeddings,
        similarity_threshold=0.25,
    )
    compressor = DocumentCompressorPipeline(
        transformers=[extractor, emb_filter]
    )
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )

    def _jsonify_md(md: Dict[str, Any]) -> Dict[str, Any]:
        clean = {}
        for k, v in (md or {}).items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                clean[k] = v
            else:
                clean[k] = str(v)
        return clean

    def _serialize_docs(docs, max_docs=20, max_chars=8000):
        out, total = [], 0
        for d in docs[:max_docs]:
            txt = d.page_content or ""
            total += len(txt)
            out.append({"page_content": txt, "metadata": _jsonify_md(d.metadata or {})})
            if total > max_chars:
                break
        return out

    def _docs_to_sources(docs) -> list[dict[str, Any]]:
        sources, seen = [], set()
        for i, d in enumerate(docs):
            md = d.metadata or {}
            sid = md.get("source") or "doc"
            page = md.get("page")
            key = (sid, page)
            if key in seen:
                continue
            seen.add(key)
            sources.append({
                "id": f"doc{i+1}",
                "title": md.get("title") or md.get("Title"),
                "page": page if isinstance(page, int) else (int(page) if str(page).isdigit() else None),
                "authors": md.get("Author"),
                "journal": md.get("Journal"),
                "publication_year": md.get("Publication Year"),
            })
        return sources

    def _join_docs(docs, max_chars: int = 10_000) -> str:
        parts, total = [], 0
        for i, d in enumerate(docs, 1):
            md = d.metadata or {}
            sid = f"doc{i}"
            page = md.get("page")
            head = f"[{i}] ({sid}{f':p.{page}' if page is not None else ''})"
            chunk = f"{head}\n{d.page_content}"
            total += len(chunk)
            parts.append(chunk)
            if total > max_chars:
                break
        return "\n\n".join(parts)

    def rag_with_sources_func(query: str) -> dict:
        docs = compression_retriever.invoke(query)
        return {
            "context": _join_docs(docs),
            "sources": _docs_to_sources(docs),
            "docs": _serialize_docs(docs)
        }

    retriever_tool = StructuredTool.from_function(
        name="retrieve_paper_chunks",
        description=(
            "Retrieve biomedical passages and structured documents for a query. "
            "Returns dict {'context': str, 'sources': List[dict], 'docs': List[{page_content, metadata}]}"
        ),
        func=rag_with_sources_func,
        args_schema=RetrieveInput,
    )
    return retriever_tool
