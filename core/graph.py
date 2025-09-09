from __future__ import annotations
from typing import Annotated, List, Tuple, Union
import operator

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END

from .prompts import build_agent_prompt, build_planner_prompt, build_replanner_prompt
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
    llm = ChatOpenAI(model=cfg.openai_chat_model, temperature=0, api_key=cfg.openai_api_key, streaming=True)

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

# ---------- Planner ----------
class Plan(BaseModel):
    """Plan to follow in future"""
    steps: list[str] = Field(description="different steps to follow, should be in sorted order")

def build_planner(llm: ChatOpenAI, tool_names: str, retriever_name: str, top_k: int = 10):
    return build_planner_prompt(tool_names, retriever_name, top_k) | llm.with_structured_output(Plan)

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
    return build_replanner_prompt(retriever_name) | llm.with_structured_output(Act)

# ---------- Graph assembly ----------
def build_app(cfg: AppConfig):
    llm = ChatOpenAI(model=cfg.openai_chat_model, temperature=0, api_key=cfg.openai_api_key, streaming=True)
    tools = build_tools(cfg)
    tool_names = ", ".join([getattr(t, "name", t.__class__.__name__) for t in tools])
    RETRIEVER_NAME = "retrieve_paper_chunks"

    agent_prompt = build_agent_prompt(tool_names, RETRIEVER_NAME)
    memory = InMemorySaver()
    agent_executor = create_react_agent(llm, tools, prompt=agent_prompt)

    planner = build_planner(llm, tool_names, RETRIEVER_NAME, top_k=cfg.top_k)
    replanner = build_replanner(llm, RETRIEVER_NAME)

    async def execute_step(state: PlanExecute, config: RunnableConfig):
        history = state.get("history", [])
        plan = state["plan"]
        plan_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))
        task = plan[0]
        available = ", ".join([getattr(t, "name", t.__class__.__name__) for t in tools])
        task_formatted = f"""For the following plan:
        {plan_str}

        Available tools: {available}

        Execute ONLY the next step:
        {task}

        STRICT RULES
        - Do not skip, reorder, or anticipate future steps.
        - If the step includes "<TOOL: name>", CALL THAT TOOL with the exact inputs specified.
        - If the step is "<THINK>", perform reasoning/synthesis using ONLY prior tool results from THIS conversation turn.
        - Never fabricate tool outputs. No external facts without a tool call this turn.
        - If you used retrieve_paper_chunks (RAG), add inline sentence-level citations exactly where each fact appears: [CIT:<id>:<page>].
        - If you used retrieve_paper_chunks (RAG), append the exact SOURCES_JSON=[...] line VERBATIM at the end of your step output.
        - NEVER add a human-readable "Sources:" section; the application renders sources.
        - If a tool fails or returns nothing, briefly state the failure and propose ONE minimal recovery action.

        RETURN FORMAT (for THIS step only)
        - Start with **Step Summary**: one–two sentences describing what you did/learned in this step.
        - Then **Reasoning**: short, structured rationale (no new external facts).
        - Then **Findings**: bullet list of results/metrics/claims. Attach [CIT:<id>:<page>] to any fact from RAG.
        - If SQL-only, present rows/aggregates succinctly under Findings (no SOURCES_JSON).
        - If RAG was used, include SOURCES_JSON=[{{...}}] as the LAST line, verbatim from the tool output.
        - Use headings to structure the answer and bold the important parts.
        """

        agent_response = await agent_executor.ainvoke({"messages": history + [("user", task_formatted)]}, config=config)
        return {
            "past_steps": [(task, agent_response["messages"][-1].content)],
            "history": [agent_response["messages"][-1]],
        }

    async def plan_step(state: PlanExecute, config: RunnableConfig):
        history = state.get("history", [])
        plan = await planner.ainvoke({"history": history, "messages": [("user", state["input"])]}, config=config)
        return {"plan": plan.steps, "history": [HumanMessage(content=state["input"])]}

    async def replan_step(state: PlanExecute, config: RunnableConfig):
        output = await replanner.ainvoke(state, config=config)
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
