from __future__ import annotations
from typing import Annotated, List, Tuple, Union
import operator

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END

from .config import AppConfig
from .retriever import build_retriever_tool

# ---------- State ----------
class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    history: Annotated[List[AnyMessage], add_messages]
    response: str

# ---------- Build tools (SQL + RAG) ----------
def build_tools(cfg: AppConfig):
    llm = ChatOpenAI(model=cfg.openai_chat_model, temperature=0, api_key=cfg.openai_api_key)

    tools = []
    # SQL tools (optional if URL is provided)
    if cfg.sql_database_url:
        sql_db = SQLDatabase.from_uri(
            cfg.sql_database_url,
            sample_rows_in_table_info=3,
            view_support=True,
            indexes_in_table_info=True,
        )
        toolkit = SQLDatabaseToolkit(db=sql_db, llm=llm)
        tools.extend(toolkit.get_tools())

    # RAG retriever tool (required for knowledge)
    retr_tool = build_retriever_tool(
        pgvector_url=cfg.pgvector_url,
        collection_name=cfg.pgvector_collection,
        openai_api_key=cfg.openai_api_key,
        openai_chat_model=cfg.openai_chat_model,
        openai_embed_model=cfg.openai_embed_model,
        top_k=cfg.top_k,
    )
    
    tools.append(retr_tool)

    return tools

# ---------- Agent system prompt (step executor) ----------
def build_agent_prompt(tool_names: str, retriever_name: str) -> str:
    return f"""
    You are a senior biomedical research assistant that solves complex questions using tools.
    You MUST use tools whenever a step requires external knowledge, structured data, calculations, or citations.
    Never fabricate tool results. Never cite facts you did not retrieve this turn.

    Available tools: {tool_names}

    === Tool Routing ===
    - Prefer <{retriever_name}> for factual/knowledge synthesis with citations.
    - Use SQL tools for structured DB lookups. Use schema-first flow:
        1) Call sql_db_list_tables to discover candidate tables.
        2) Call sql_db_schema with the SPECIFIC table names you intend to query (include exact names).
        3) Draft SQL text; then call sql_db_query_checker to validate/fix SQL.
        4) Execute only with sql_db_query using the corrected SQL.
        Never call sql_db_query before steps (1)–(3). If uncertain, re-check schema.

    === {retriever_name} Usage ===
    - Returns a single string with two sections:
    CONTEXT: <chunked passages>
    SOURCES_JSON=[{{"title": "...", "page": ..., "authors": "...", "journal": "...",
                        "publication_year": ..., "population_flag": ..., "intervention_flag": ...,
                        "comparison_flag": ..., "outcome_flag": ..., "study_design_flag": ...,
                        "PICOS_qualified_flag": "..."}}...]
    - REQUIRED actions when you use {retriever_name} in a step:
    a) Write a concise, factual summary using only the CONTEXT.
    b) Insert inline citations as [CIT:<title_or_shortid>:<page>] when referencing CONTEXT facts.
    c) Append the SOURCES_JSON=[...] line VERBATIM at the end of your step output.
    - Do NOT prepend answers with “Source(s)”. Keep citations inline as specified.

    === Step Execution Rules ===
    1) Follow the current plan step exactly; execute ONLY that step.
    2) If a step requires a tool, CALL IT. Do not infer tool outputs.
    3) If a tool fails or returns nothing:
        - State the failure briefly and propose ONE minimal recovery (e.g., broaden query terms or inspect additional tables).
    4) Be concise, neutral, and structured; your outputs feed later steps.
    5) If you did NOT call any tool in this step, you may NOT introduce new external facts or citations.

    === Output Style (PER STEP) ===
    - Start with a concise summary of what you did/learned in THIS step.
    - Include inline citations only for facts from CONTEXT (format: [CIT:...:...]).
    - If you called {retriever_name}, append the exact SOURCES_JSON=[...] line at the bottom.
    - If the step was SQL-only, do NOT include SOURCES_JSON; instead, present the JSON rows/aggregate succinctly.
    - Never invent metadata not present in tool outputs.
    
    === Examples (abbreviated) ===
    (RET) Summary: Identified adjuvant options for stage II disease [CIT:Smith_2023:2].
    SOURCES_JSON=[{{"title":"Smith 2023","page":2,"authors":"...","journal":"...","publication_year":2023,...}}]

    (SQL) Summary: Counted eligible trials by year (2018–2024) via table trials → 127 total.
    Rows: [{{"year":2018,"n":12}}, {{"year":2019,"n":18}}, ...]

    Compliance notes:
    - Do not start your output with “Source(s)”.
    - No external facts without a tool call this turn.
    """

# ---------- Planner ----------
class Plan(BaseModel):
    """Plan to follow in future"""
    steps: list[str] = Field(description="different steps to follow, should be in sorted order")


def build_planner(llm: ChatOpenAI, tool_names: str, retriever_name: str, top_k: int = 10):
    planner_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a planner for a tool-using biomedical assistant. Produce a **minimal, executable plan** to achieve the user's objective using ONLY the available tools.

                AVAILABLE TOOLS:
                - {tool_names}

                KEY TOOL POLICIES:
                - **Retrieval-first**: If the objective needs external knowledge, the **first** retrieval step MUST use <TOOL: {retriever_name}> to fetch top-{top_k} passages with recency bias.
                - **SQL schema-first**: If any SQL is needed, you MUST:
                1) <TOOL: sql_db_list_tables> to discover candidates,
                2) <TOOL: sql_db_schema> for the **specific tables** you intend to query (include exact table names),
                3) Draft SQL (as text) and validate via <TOOL: sql_db_query_checker>,
                4) Execute with <TOOL: sql_db_query>.
                Never execute SQL without steps (1)–(3) in this order.

                PLANNING RULES:
                - Each step is a **single atomic action** with concrete inputs and a **measurable success** signal.
                - Prefer the fewest steps that can succeed reliably.
                - If the question might be answerable from context alone, you may use <THINK>, but **default to retrieval** when in doubt.
                - If tools were used, the **final step must produce the final answer with inline citations** (e.g., “[D12]”, “[T.p3]”) drawn from retrieved docs or query results. **Do not begin the answer with “Source(s)”**.

                STEP FORMAT (one line per step; no extra commentary, no numbering prefix like “1)”):
                - Start with a tag: **<TOOL: exact_tool_name>** when calling a tool, or **<THINK>** for reasoning-only steps.
                - Follow with a short, imperative action.
                - If a tool is called, include the **exact input arguments** you will pass.
                - End with: **| success: <observable success criterion>**

                EXAMPLES:
                <TOOL: {retriever_name}> Retrieve top-{top_k} passages about "stage II breast cancer options" with recency bias | success: passages returned with doc IDs
                <THINK> Select 3 non-duplicative, highest-authority passages covering options and contraindications | success: 3 passage IDs chosen with 1-line rationale
                <TOOL: sql_db_list_tables> List tables | success: tables listed
                <TOOL: sql_db_schema> tables=["patients","oncology_events"] | success: columns, types, and 3 sample rows per table returned
                <THINK> Draft SELECT with JOIN patients.id=oncology_events.patient_id WHERE patient_id=42 (age, comorbidities, prior_radiation) | success: syntactically correct draft SQL
                <TOOL: sql_db_query_checker> sql="SELECT p.age, p.comorbidities, e.prior_radiation FROM patients p JOIN oncology_events e ON p.id=e.patient_id WHERE p.id=42" | success: checker returns OK or a corrected SQL
                <TOOL: sql_db_query> sql="<corrected SQL>" | success: JSON row returned
                <THINK> Synthesize the answer integrating clinical facts and retrieved evidence with inline citations like [D12], [D7.p2] | success: final draft ready with citations

                CONSTRAINTS:
                - No superfluous steps. Every step must be independently executable by the agent.
                - When using SQL, **schema steps are mandatory before execution**.
                - The final step MUST output the final answer (with citations if tools were used).

                RETURN:
                Output **only** the ordered list of steps in the specified line format. No headers, no prose, no numbering characters.
                """,
            ),
            ("placeholder", "{history}"),
            ("placeholder", "{messages}"),
        ]
    )
    return planner_prompt.partial(tool_names=tool_names, retriever_name=retriever_name, top_k=top_k) | llm.with_structured_output(Plan)

# ---------- Replanner / Decider ----------
class Response(BaseModel):
    """Response to user."""
    response: str

class Act(BaseModel):
    """Action to perform."""
    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use **Response**. "
                    "If you need to further use tools to get the answer, use **Plan**."
    )

def build_replanner(llm: ChatOpenAI, retriever_name: str):
    replanner_prompt = ChatPromptTemplate.from_template(
    """
    You are updating a tool-aware plan mid-execution.

    Objective:
    {input}

    Original plan:
    {plan}

    Steps completed (with outcomes):
    {past_steps}

    Rules:
    - If information is still missing, ADD the minimal next steps needed.
    - Prefer using the retrieve_paper_chunks tool FIRST for missing knowledge.
    - Keep steps executable and specific, using the same step FORMAT as the planner:
    "<TOOL: name> Action with exact inputs | success: ...", or "<THINK> ... | success: ..."
    - Do NOT repeat already-completed steps.

    Finalization:
    - When all needed info is obtained, return a final Response that:
    1) Answers the user succinctly and cites inline as [CIT:<id>:<page>] where appropriate.
    2) Compiles a deduplicated 'Sources' list by scanning all prior step outputs for
        a literal block like: SOURCES_JSON=[{{...}}, ...]
        - Each item should include any available fields among: id/source/document_id, title, page, url.
        - Deduplicate by (id, page) if both exist; otherwise by id.
    3) If no SOURCES_JSON blocks exist, return the answer and say "Sources: (none)".

    Decide:
    - If more tool use is required → return Plan with ONLY the remaining steps.
    - If the answer is ready → return Response (final answer with a 'Sources:' section).
    """
    )
    return replanner_prompt | llm.with_structured_output(Act)

# ---------- Graph assembly ----------
def build_app(cfg: AppConfig):
    llm = ChatOpenAI(model=cfg.openai_chat_model, temperature=0, api_key=cfg.openai_api_key)
    tools = build_tools(cfg)
    tool_names = ", ".join([getattr(t, "name", t.__class__.__name__) for t in tools])
    RETRIEVER_NAME = "retrieve_paper_chunks"

    agent_prompt = build_agent_prompt(tool_names, RETRIEVER_NAME)
    memory = InMemorySaver()
    agent_executor = create_react_agent(llm, tools, prompt=agent_prompt)

    planner = build_planner(llm, tool_names, RETRIEVER_NAME, top_k=cfg.top_k)
    replanner = build_replanner(llm, RETRIEVER_NAME)

    async def execute_step(state: PlanExecute):
        history = state.get("history", [])
        plan = state["plan"]
        plan_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))
        task = plan[0]
        available = ", ".join([getattr(t, "name", t.__class__.__name__) for t in tools])
        task_formatted = f"""For the following plan:
            {plan_str}

            Available tools: {available}

            Execute ONLY the next step: {task}
            - If the step includes <TOOL: name>, call that tool with the specified input.
            - If the step is <THINK>, do the reasoning/synthesis using prior tool results.
            - When done, return a concise result for this step (include citations if from RAG/search).
        """
        agent_response = await agent_executor.ainvoke({"messages": history + [("user", task_formatted)]})
        return {
            "past_steps": [(task, agent_response["messages"][-1].content)],
            "history": [agent_response["messages"][-1]],
        }

    async def plan_step(state: PlanExecute):
        history = state.get("history", [])
        plan = await planner.ainvoke({"history": history, "messages": [("user", state["input"])]})
        return {"plan": plan.steps, "history": [HumanMessage(content=state["input"])]}

    async def replan_step(state: PlanExecute):
        output = await replanner.ainvoke(state)
        if isinstance(output.action, Response):
            return {
                "response": output.action.response,
                "history": [AIMessage(content=output.action.response)],
                "plan": [],
            }
        else:
            return {"plan": output.action.steps}

    def should_end(state: PlanExecute):
        has_answer = bool(state.get("response"))
        has_plan = bool(state.get("plan"))
        return END if (has_answer and not has_plan) else "agent"

    workflow = StateGraph(PlanExecute)
    workflow.add_node("planner", plan_step)
    workflow.add_node("agent", execute_step)
    workflow.add_node("replan", replan_step)
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "agent")
    workflow.add_edge("agent", "replan")
    workflow.add_conditional_edges("replan", should_end, ["agent", END])
    app = workflow.compile(checkpointer=memory)
    return app
