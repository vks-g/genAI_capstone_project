"""
LangGraph workflow definition for the Churn Retention Agent.
Compiles the StateGraph from the four node functions into a runnable agent.
"""

import os

from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import END, StateGraph
from src.agent.state import ChurnAgentState
from src.agent.nodes import (
    assess_risk,
    generate_report,
    plan_intervention,
    retrieve_strategies,
)


def _should_continue_after_planning(state: ChurnAgentState) -> str:
    """
    Conditional edge: if plan_intervention set a fatal error, skip to report
    generation anyway so the user always gets a structured response.
    LangGraph requires this to return the name of the next node as a string.
    """
    return "generate_report"


def build_retention_agent():
    """
    Build and compile the LangGraph StateGraph.
    Returns the compiled runnable agent.
    """
    workflow = StateGraph(ChurnAgentState)

    # Register nodes
    workflow.add_node("assess_risk", assess_risk)
    workflow.add_node("retrieve_strategies", retrieve_strategies)
    workflow.add_node("plan_intervention", plan_intervention)
    workflow.add_node("generate_report", generate_report)

    # Define the linear flow
    workflow.set_entry_point("assess_risk")
    workflow.add_edge("assess_risk", "retrieve_strategies")
    workflow.add_edge("retrieve_strategies", "plan_intervention")

    # Conditional edge after planning (always goes to report, handles errors internally)
    workflow.add_conditional_edges(
        "plan_intervention",
        _should_continue_after_planning,
        {"generate_report": "generate_report"},
    )

    workflow.add_edge("generate_report", END)

    return workflow.compile()


# Module-level compiled agent - imported by app.py
# Compiled once at import time to avoid re-compilation on each Streamlit rerun
retention_agent = build_retention_agent()
