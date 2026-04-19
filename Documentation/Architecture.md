# Agent Workflow & System Architecture

**Customer Churn Prediction & Agentic Retention Strategy**
*GenAI Capstone Project — Project 5*

> Live Demo: https://genaicapstoneproject.streamlit.app/
> GitHub: https://github.com/Shreyashgol/genAI_capstone_project

---

## Overview

This project is built in two progressive milestones that together form a complete AI-driven customer analytics system.

**Milestone 1** is a classical ML pipeline that answers: *which customers will churn?*

**Milestone 2** is a LangGraph-powered agentic system that answers: *what should we do about it?*

The two milestones are tightly integrated — the churn probability score produced by Milestone 1 seeds the initial state of the Milestone 2 agent pipeline.

---

## Grand Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              Telco Customer Churn Dataset                       │
│        7,043 customers · 21 columns · CSV (Kaggle/IBM)          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  MILESTONE 1 — ML Pipeline                      │
│                                                                 │
│  [Preprocessing] ──► [Feature Engineering] ──► [SMOTE]         │
│   · Drop customerID    · TenureGroup              · Train only  │
│   · Fix TotalCharges   · ChargeRatio              · 80/20 split │
│   · One-hot encode     · StandardScaler                         │
│                                                                 │
│          ┌──────────────────────────────────────┐               │
│          │      Model Training (parallel)       │               │
│          │                                      │               │
│   [Logistic Regression]  [Decision Tree]  [Random Forest]       │
│    Acc: 79.35%            Acc: 77.15%      Acc: 79.28%          │
│    F1:  63.49%            F1:  62.56%      F1:  64.39%          │
│                                                                 │
│          └──────────────┬───────────────────────┘               │
│                         ▼                                       │
│                  [Evaluation]                                   │
│         Accuracy · Precision · Recall · F1 · Confusion Matrix   │
│                         │                                       │
│                         ▼                                       │
│            [Model Serialisation — joblib]                       │
│    logistic_regression_model.pkl                                │
│    decision_tree_model.pkl                                      │
│    random_forest_model.pkl                                      │
│    scaler.pkl · model_columns.pkl · test_data.pkl               │
└────────────────────────┬────────────────────────────────────────┘
                         │  churn_probability (float 0–1)
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│           MILESTONE 2 — LangGraph Agentic Pipeline              │
│                                                                 │
│  ChurnAgentState (TypedDict) flows through all nodes:           │
│  customer_data · churn_probability · risk_level                 │
│  risk_drivers · retrieval_query · retrieved_strategies          │
│  llm_reasoning · retention_report · error                       │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Node 1 — assess_risk                                     │   │
│  │  · Classify probability into low / medium / high tier   │   │
│  │  · Extract human-readable risk drivers from features    │   │
│  │  · Drivers: contract type, tenure, internet, security,  │   │
│  │    tech support, payment method, monthly charges,       │   │
│  │    senior citizen status, partner/dependents            │   │
│  └──────────────────────────┬───────────────────────────────┘   │
│                             │                                   │
│                             ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Node 2 — retrieve_strategies  (RAG)                      │   │
│  │  · Build semantic query from risk tier + drivers         │   │
│  │  · Query Chroma vector store (similarity search, k=4)   │   │
│  │  · Return top-4 relevant text chunks                    │   │
│  │                                                          │   │
│  │  Knowledge Base ──────────────────────────────────────   │   │
│  │  · 7 industry PDFs (customer success, loyalty, CRM)     │   │
│  │  · ~2,000 chunks (400 tokens, 50 overlap)               │   │
│  │  · Embedded with intfloat/e5-small (HuggingFace)        │   │
│  │  · Persisted to chroma_db/ (committed to repo)          │   │
│  └──────────────────────────┬───────────────────────────────┘   │
│                             │                                   │
│                             ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Node 3 — plan_intervention  (Groq LLM)                   │   │
│  │  · Model: llama-3.1-8b-instant via Groq API             │   │
│  │  · Temperature: 0.3 (consistent, factual output)        │   │
│  │  · Input: customer profile + risk drivers + RAG chunks  │   │
│  │  · Output: strict JSON with 3 keys:                     │   │
│  │      "risk_summary"        — 2-3 sentence analysis      │   │
│  │      "recommended_actions" — list of 3-5 actions        │   │
│  │      "reasoning"           — why these actions fit      │   │
│  │  · Anti-hallucination: grounded-only instruction,       │   │
│  │    JSON-only output, low temperature                     │   │
│  └──────────────────────────┬───────────────────────────────┘   │
│                             │                                   │
│                             ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Node 4 — generate_report                                 │   │
│  │  · Parse LLM JSON with 3-level fallback:                │   │
│  │      1. Direct json.loads()                             │   │
│  │      2. Strip markdown fences, retry                    │   │
│  │      3. Heuristic {…} extraction                        │   │
│  │      4. Safe fallback report (always returns something) │   │
│  │  · Append knowledge base source previews               │   │
│  │  · Append hardcoded ethical disclaimer (GDPR/CCPA)     │   │
│  └──────────────────────────┬───────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │  retention_report (dict)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              Streamlit UI — 5-page Application                  │
│                                                                 │
│  Page 1: Intro         Project overview + key features          │
│  Page 2: Input         19-field customer data form              │
│  Page 3: Model select  Logistic Reg / Decision Tree / RF        │
│  Page 4: Results       Gauge · feature importance · metrics     │
│                        confusion matrix · "Run Agent" button    │
│  Page 5: Agent report  Risk badge · risk summary · drivers      │
│                        recommended actions · reasoning          │
│                        knowledge base sources · disclaimer      │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│           Streamlit Community Cloud (Deployment)                │
│         https://genaicapstoneproject.streamlit.app/             │
│                                                                 │
│  · Python 3.12 · API key in Streamlit Secrets                   │
│  · chroma_db/ committed to repo (no rebuild needed)             │
│  · Continuous deployment on GitHub push                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Node-by-Node Reference

### Node 1 — `assess_risk`

**File:** `src/agent/nodes.py`

**Input state fields used:**
- `churn_probability` — float from ML model
- `customer_data` — raw dict of 19 customer features

**Processing:**

Classifies the churn probability into a risk tier using fixed thresholds:
- High risk: probability ≥ 65%
- Medium risk: 35% ≤ probability < 65%
- Low risk: probability < 35%

Then maps raw feature values to human-readable risk driver strings. Checked features and their corresponding drivers:

| Feature | Condition | Driver string |
|---|---|---|
| Contract | Month-to-month | "month-to-month contract (no long-term commitment)" |
| tenure | < 12 | "very short tenure (N months)" |
| tenure | 12–23 | "short tenure (N months)" |
| InternetService | Fiber optic | "fiber optic service (higher churn segment)" |
| OnlineSecurity | No | "no online security add-on" |
| TechSupport | No | "no tech support subscription" |
| PaymentMethod | Electronic check | "electronic check payment (lower engagement signal)" |
| MonthlyCharges | > $70 | "high monthly charges ($N)" |
| SeniorCitizen | 1 | "senior citizen (higher churn demographic)" |
| Partner + Dependents | Both No | "no partner or dependents (lower switching cost)" |

**Output state fields set:**
- `risk_level` — "low", "medium", or "high"
- `risk_drivers` — list of driver strings

---

### Node 2 — `retrieve_strategies`

**File:** `src/agent/nodes.py`, `src/agent/retriever.py`

**Input state fields used:**
- `risk_level`, `risk_drivers`, `customer_data`

**Processing:**

Constructs a targeted semantic search query:
```
"customer retention strategy for {risk_level} churn risk customer
with {contract} contract, {tenure} months tenure,
{internet} internet service. Key issues: {driver1}, {driver2}"
```

Queries the Chroma vector store using cosine similarity with `k=4`. The vector store is backed by `intfloat/e5-small` embeddings and contains ~2,000 chunks from 7 industry PDFs.

Falls back gracefully to an empty list if Chroma is unavailable, allowing the pipeline to continue with a general LLM response.

**Knowledge base documents:**
1. `01_customer-revenue-study-2025.pdf`
2. `02_CSM Confidential 2025.pdf`
3. `03_stragatic_research.pdf`
4. `04_loyalty-and-customer-satisfaction.pdf`
5. `05_strategic_research.pdf`
6. `06.customersuccessplaybookinteractive.pdf`
7. `07_The-Customer-360-Playbook.pdf`

**Output state fields set:**
- `retrieval_query` — the constructed query string
- `retrieved_strategies` — list of up to 4 text chunks

---

### Node 3 — `plan_intervention`

**File:** `src/agent/nodes.py`, `src/agent/prompts.py`

**Input state fields used:**
- `churn_probability`, `risk_level`, `risk_drivers`, `customer_data`, `retrieved_strategies`

**LLM configuration:**
- Provider: Groq API (free tier)
- Model: `llama-3.1-8b-instant`
- Temperature: 0.3
- Max tokens: 1024
- API key: resolved from Streamlit Secrets → `.env` fallback

**Hallucination mitigation techniques:**
1. System prompt explicitly forbids recommendations not grounded in retrieved knowledge
2. All retrieved chunks are injected into the user prompt
3. Strict JSON-only output instruction — no preamble, no markdown fences
4. Low temperature (0.3) reduces creative deviation
5. Exact JSON schema specified in system prompt

**Output schema (enforced by prompt):**
```json
{
  "risk_summary": "2-3 sentence explanation of why customer is at risk",
  "recommended_actions": ["action 1", "action 2", "action 3"],
  "reasoning": "2-3 sentences explaining why these actions fit this customer"
}
```

**Output state fields set:**
- `llm_reasoning` — raw LLM response string

---

### Node 4 — `generate_report`

**File:** `src/agent/nodes.py`

**Input state fields used:**
- `llm_reasoning`, `retrieved_strategies`, `risk_level`, `churn_probability`, `error`

**Processing:**

Parses the LLM output with cascading fallback:
1. Direct `json.loads()` on raw output
2. Strip markdown code fences and retry
3. Extract substring between first `{` and last `}` and retry
4. If all fail: return a safe structured report with standard best-practice actions

Appends the first 120 characters of each retrieved chunk as source citations.

Always appends the ethical disclaimer regardless of whether LLM parsing succeeded.

**Ethical disclaimer (hardcoded):**
> This retention strategy was generated by an AI system and is intended as decision-support for human agents, not as an autonomous decision-making tool. Recommendations should be reviewed by a qualified customer success professional before any action is taken. Customer data used in this analysis must be handled in accordance with applicable data protection regulations (e.g., GDPR, CCPA). The probability score is a statistical estimate and does not guarantee future customer behaviour.

**Output state fields set:**
- `retention_report` — dict with keys: `risk_summary`, `recommended_actions`, `reasoning`, `sources`, `ethical_disclaimer`

---

## State Object Reference

**File:** `src/agent/state.py`

```python
class ChurnAgentState(TypedDict):
    customer_data: dict         # Raw 19-field customer dict from Streamlit form
    churn_probability: float    # ML model output (0.0 to 1.0)
    risk_level: str             # "low", "medium", or "high"
    risk_drivers: list          # Human-readable risk signals
    retrieval_query: str        # Query sent to Chroma
    retrieved_strategies: list  # Text chunks from knowledge base
    llm_reasoning: str          # Raw LLM JSON string
    retention_report: dict      # Final structured report
    error: Optional[str]        # Set if any node fails
```

---

## Graph Compilation

**File:** `src/agent/graph.py`

```
assess_risk
    │
    ▼
retrieve_strategies
    │
    ▼
plan_intervention
    │  (conditional edge — always routes to generate_report)
    ▼
generate_report
    │
    ▼
END
```

The conditional edge after `plan_intervention` ensures the graph always reaches `generate_report` even if the LLM call fails, producing a safe fallback report in all cases rather than crashing the pipeline.

The compiled agent (`retention_agent`) is instantiated once at module import time to avoid recompilation on every Streamlit rerun.

---

## Technology Stack

| Layer | Component | Technology |
|---|---|---|
| Data | Dataset | Telco Customer Churn (Kaggle/IBM) |
| ML | Models | Logistic Regression, Decision Tree, Random Forest |
| ML | Preprocessing | pandas, NumPy, scikit-learn 1.6.1 |
| ML | Balancing | imbalanced-learn (SMOTE) |
| ML | Serialisation | joblib (.pkl) |
| Agent | Framework | LangGraph (StateGraph) |
| Agent | LLM | Groq API — llama-3.1-8b-instant |
| Agent | RAG | Chroma + langchain-chroma |
| Agent | Embeddings | intfloat/e5-small (sentence-transformers) |
| Agent | Document load | PyPDFLoader (langchain-community) |
| UI | Framework | Streamlit |
| UI | Charts | Plotly |
| Deploy | Hosting | Streamlit Community Cloud |
| Deploy | Runtime | Python 3.12 |
| Deploy | Secrets | Streamlit Secrets (GROQ_API_KEY) |

---

## Repository Structure

```
genAI_capstone_project/
├── app.py                          # 5-page Streamlit application
├── requirements.txt                # Pinned dependencies
├── .env.example                    # Environment variable template
├── README.md
├── data/
│   └── Telco-Customer-Churn.csv    # 7,043 customer records
├── models/
│   ├── logistic_regression_model.pkl
│   ├── decision_tree_model.pkl
│   ├── random_forest_model.pkl
│   ├── scaler.pkl
│   ├── model_columns.pkl
│   ├── test_data.pkl               # Pre-saved test split
│   └── metrics.json                # Pre-computed scores
├── knowledge_base/                 # 7 industry PDFs
├── chroma_db/                      # Pre-built Chroma vector store
├── notebook/
│   └── Telco_Customer_Churn.ipynb  # EDA + training notebook
├── src/
│   ├── preprocessing.py            # Inference preprocessing
│   ├── model_training.py           # Model loader
│   ├── evaluation.py               # Metrics helpers
│   └── agent/
│       ├── state.py                # ChurnAgentState definition
│       ├── nodes.py                # 4 LangGraph node functions
│       ├── graph.py                # StateGraph compilation
│       ├── prompts.py              # LLM prompt templates
│       ├── retriever.py            # Chroma retrieval
│       ├── embedder.py             # Vector store builder
│       ├── document_loader.py      # PDF chunker
│       ├── build_vectorstore.py    # One-time builder script
│       └── test_agent.py           # Integration test
└── Documentation/
    └── Report.pdf
```

---

## Running Locally

```bash
# 1. Clone and install
git clone https://github.com/vks-g/genAI_capstone_project
cd genAI_capstone_project
pip install -r requirements.txt

# 2. Set environment variables
cp .env.example .env
# Add your GROQ_API_KEY to .env

# 3. The chroma_db/ is already built — no rebuild needed
# If you add new PDFs to knowledge_base/, rebuild with:
python -m src.agent.build_vectorstore

# 4. Run the app
streamlit run app.py

# 5. Test the agent pipeline end-to-end
python -m src.agent.test_agent
```

---

## Team

| Name | Enrolment No. |
|---|---|
| Gokul VKS | 2401020094 |
| Shreyash Golhani | 2401020069 |
| Vaageesh Kumar Singh | 2401020073 |
| Mohammad Affan Anas | 2401010280 |

---

*Submitted for End-Semester Evaluation — GenAI Capstone Course, 2026*
