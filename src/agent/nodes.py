"""
LangGraph node functions for the Churn Retention Agent.
Each function represents one step in the reasoning pipeline.
"""

import os
from src.agent.state import ChurnAgentState


def assess_risk(state: ChurnAgentState) -> ChurnAgentState:
    """
    Node 1: Risk Assessment
    Classifies churn_probability into risk_level (low/medium/high).
    Extracts human-readable risk_drivers from customer_data field values.
    """
    prob = state["churn_probability"]
    data = state["customer_data"]

    # Classify risk tier
    if prob >= 0.65:
        risk_level = "high"
    elif prob >= 0.35:
        risk_level = "medium"
    else:
        risk_level = "low"

    # Map raw customer features to human-readable risk signals
    drivers = []

    contract = data.get("Contract", "")
    if contract == "Month-to-month":
        drivers.append("month-to-month contract (no long-term commitment)")
    elif contract == "One year":
        drivers.append("one-year contract (moderate commitment)")

    tenure = data.get("tenure", 0)
    if tenure < 12:
        drivers.append(f"very short tenure ({tenure} months)")
    elif tenure < 24:
        drivers.append(f"short tenure ({tenure} months)")

    internet = data.get("InternetService", "")
    if internet == "Fiber optic":
        drivers.append("fiber optic service (higher churn segment)")

    security = data.get("OnlineSecurity", "")
    if security == "No":
        drivers.append("no online security add-on")

    tech = data.get("TechSupport", "")
    if tech == "No":
        drivers.append("no tech support subscription")

    payment = data.get("PaymentMethod", "")
    if payment == "Electronic check":
        drivers.append("electronic check payment (lower engagement signal)")

    monthly = data.get("MonthlyCharges", 0)
    if monthly > 70:
        drivers.append(f"high monthly charges (${monthly:.0f})")

    senior = data.get("SeniorCitizen", 0)
    if senior == 1:
        drivers.append("senior citizen (higher churn demographic)")

    partner = data.get("Partner", "No")
    dependents = data.get("Dependents", "No")
    if partner == "No" and dependents == "No":
        drivers.append("no partner or dependents (lower switching cost)")

    paperless = data.get("PaperlessBilling", "No")
    if paperless == "Yes":
        drivers.append("paperless billing enabled")

    # Guarantee at least one driver even for edge cases
    if not drivers:
        drivers.append(f"churn probability of {prob:.1%} detected by ML model")

    return {
        **state,
        "risk_level": risk_level,
        "risk_drivers": drivers,
    }


def retrieve_strategies(state: ChurnAgentState) -> ChurnAgentState:
    """
    Node 2: RAG Retrieval
    Builds a retrieval_query from risk_level and risk_drivers.
    Calls retriever.retrieve_strategies() to fetch relevant chunks.
    Returns updated state with retrieval_query and retrieved_strategies populated.
    """
    raise NotImplementedError("To be implemented in agent implementation prompt.")


def plan_intervention(state: ChurnAgentState) -> ChurnAgentState:
    """
    Node 3: LLM-Based Planning
    Constructs a prompt from customer profile + retrieved strategy chunks.
    Calls Groq LLM via LangChain to generate retention recommendations.
    Returns updated state with llm_reasoning populated.
    """
    raise NotImplementedError("To be implemented in agent implementation prompt.")


def generate_report(state: ChurnAgentState) -> ChurnAgentState:
    """
    Node 4: Report Generation
    Structures llm_reasoning into a clean retention_report dict.
    Format: { risk_summary, recommended_actions, sources, ethical_disclaimer }
    Returns updated state with retention_report populated.
    """
    raise NotImplementedError("To be implemented in agent implementation prompt.")
