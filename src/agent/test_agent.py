"""
End-to-end integration test for the Churn Retention Agent.
Run from repo root: python -m src.agent.test_agent
Tests the full pipeline: assess_risk -> retrieve -> plan -> report
"""

import json

from dotenv import load_dotenv

load_dotenv()

from src.agent.graph import retention_agent

# --- Test Case 1: High Risk Customer ---
HIGH_RISK_CUSTOMER = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 3,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.0,
    "TotalCharges": 255.0,
}

# --- Test Case 2: Low Risk Customer ---
LOW_RISK_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "Yes",
    "tenure": 48,
    "PhoneService": "Yes",
    "MultipleLines": "Yes",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "Yes",
    "DeviceProtection": "Yes",
    "TechSupport": "Yes",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Two year",
    "PaperlessBilling": "No",
    "PaymentMethod": "Bank transfer (automatic)",
    "MonthlyCharges": 55.0,
    "TotalCharges": 2640.0,
}


def run_test(label: str, customer_data: dict, churn_prob: float):
    print(f"\n{'=' * 60}")
    print(f"TEST: {label}")
    print(f"{'=' * 60}")

    initial_state = {
        "customer_data": customer_data,
        "churn_probability": churn_prob,
        "risk_level": "",
        "risk_drivers": [],
        "retrieval_query": "",
        "retrieved_strategies": [],
        "llm_reasoning": "",
        "retention_report": {},
        "error": None,
    }

    result = retention_agent.invoke(initial_state)

    print(f"Risk Level:      {result['risk_level']}")
    print(f"Risk Drivers:    {result['risk_drivers']}")
    print(f"Retrieval Query: {result['retrieval_query']}")
    print(f"Chunks Retrieved:{len(result['retrieved_strategies'])}")

    report = result["retention_report"]
    print("\n--- RETENTION REPORT ---")
    print(f"Risk Summary: {report.get('risk_summary', 'N/A')}")
    print("\nRecommended Actions:")
    for i, action in enumerate(report.get("recommended_actions", []), 1):
        print(f"  {i}. {action}")
    print(f"\nSources: {len(report.get('sources', []))} references")
    print(f"\nDisclaimer present: {'ethical_disclaimer' in report}")

    if result.get("error"):
        print(f"\n⚠️  Agent Error: {result['error']}")

    return result


if __name__ == "__main__":
    print("Running Churn Retention Agent Integration Tests...")

    r1 = run_test("High Risk Customer", HIGH_RISK_CUSTOMER, churn_prob=0.82)
    assert r1["risk_level"] == "high", f"Expected high, got {r1['risk_level']}"
    assert len(r1["retention_report"].get("recommended_actions", [])) >= 1

    r2 = run_test("Low Risk Customer", LOW_RISK_CUSTOMER, churn_prob=0.11)
    assert r2["risk_level"] == "low", f"Expected low, got {r2['risk_level']}"
    assert "ethical_disclaimer" in r2["retention_report"]

    print(f"\n{'=' * 60}")
    print("✅ All integration tests passed.")
    print(f"{'=' * 60}")
