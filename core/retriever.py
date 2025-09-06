from __future__ import annotations
from typing import List, Dict, Any
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_community.vectorstores import PGVector
# from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.retrievers import SelfQueryRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain.tools import Tool

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
) -> Tool:
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
        search_kwargs={"k": top_k},
    )

    # 2) LLM-based filter as contextual compression
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=LLMChainFilter.from_llm(llm),
        base_retriever=base_retriever,
    )

    def _docs_to_sources(docs) -> list[dict[str, Any]]:
        sources = []
        seen = set()
        for i, d in enumerate(docs):
            md = d.metadata or {}
            sid = md.get("source") or "doc"
            page = md.get("page")
            title = md.get("title") or md.get("Title")
            authors = md.get("Author")
            journal = md.get("Journal")
            year = md.get("Publication Year")
            population_flag = md.get("Population")
            intervention_flag = md.get("Intervention")
            comparison_flag = md.get("Comparator")
            outcome_flag = md.get("Outcome")
            study_design_flag = md.get("Study Design")
            PICOS_qualified_flag = md.get("Qualification")
            key = (sid, page)
            if key in seen:
                continue
            seen.add(key)
            sources.append({
                "id": f"doc{i+1}",
                "title": title,
                "page": page if isinstance(page, int) else (int(page) if str(page).isdigit() else None),
                "authors": authors,
                "journal": journal,
                "publication_year": year,
                "population_flag": population_flag,
                "intervention_flag": intervention_flag,
                "comparison_flag": comparison_flag,
                "outcome_flag": outcome_flag,
                "study_design_flag": study_design_flag,
                "PICOS_qualified_flag": PICOS_qualified_flag,
            })
        return sources

    def _join_docs(docs, max_chars: int = 10_000) -> str:
        parts = []
        for i, d in enumerate(docs, 1):
            md = d.metadata or {}
            sid = f"doc{i}"
            page = md.get("page")
            head = f"[{i}] ({sid}{f':p.{page}' if page is not None else ''})"
            parts.append(f"{head}\n{d.page_content}")
            if sum(len(p) for p in parts) > max_chars:
                break
        return "\n\n".join(parts)

    def rag_with_sources_func(query: str) -> str:
        docs = compression_retriever.invoke(query)
        sources = _docs_to_sources(docs)
        context = _join_docs(docs)
        import json
        sources_json = json.dumps(sources, ensure_ascii=False)
        return f"CONTEXT:\n{context}\n\nSOURCES_JSON={sources_json}"

    retriever_tool = Tool(
        name="retrieve_paper_chunks",
        description="Search and return information about medical research papers, always emits SOURCES_JSON first.",
        func=rag_with_sources_func,
    )
    return retriever_tool
