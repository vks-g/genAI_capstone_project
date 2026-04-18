import streamlit as st
import pandas as pd
import numpy as np
import json
import pickle
import plotly.graph_objects as go
import plotly.express as px
from src.model_training import load_all_models
from src.preprocessing import preprocess_input, load_columns, load_scaler

try:
    from src.agent.graph import retention_agent

    AGENT_AVAILABLE = True
except Exception as _agent_import_err:
    retention_agent = None
    AGENT_AVAILABLE = False
    print(f"[app.py] Agent import failed: {_agent_import_err}")

st.set_page_config(
    page_title="Customer Churn Prediction AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
    :root {
        --panel-glass: rgba(255, 255, 255, 0.92);
        --panel-border: rgba(255, 255, 255, 0.26);
        --panel-shadow: 0 20px 60px rgba(8, 18, 44, 0.34);
        --text-soft: #e5e7eb;
    }

    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #7e22ce 100%);
    }
    
    [data-testid="stSidebar"] {
        display: none;
    }
    
    .card {
        background: var(--panel-glass);
        border: 1px solid var(--panel-border);
        backdrop-filter: blur(6px);
        padding: 40px;
        border-radius: 20px;
        box-shadow: var(--panel-shadow);
        margin: 30px 0;
        transition: all 0.4s ease;
    }
    
    .card:hover {
        transform: translateY(-8px);
        box-shadow: 0 25px 70px rgba(0,0,0,0.4);
    }
    
    .model-card {
        background: linear-gradient(135deg, #ffffff 0%, #f0f4ff 100%);
        padding: 35px;
        border-radius: 20px;
        text-align: center;
        cursor: pointer;
        border: 4px solid transparent;
        transition: all 0.4s ease;
        min-height: 280px;
    }
    
    .model-card:hover {
        border-color: #7e22ce;
        transform: scale(1.08);
        box-shadow: 0 15px 40px rgba(126, 34, 206, 0.3);
    }
    
    .main-title {
        font-size: 4em;
        font-weight: 900;
        color: white;
        text-align: center;
        margin-bottom: 20px;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.4);
    }
    
    .subtitle {
        font-size: 1.4em;
        color: #e0e7ff;
        text-align: center;
        margin-bottom: 50px;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #7e22ce 0%, #ec4899 100%);
        color: white;
        border: none;
        padding: 18px 50px;
        font-size: 20px;
        font-weight: 700;
        border-radius: 50px;
        transition: all 0.4s ease;
    }
    
    .stButton>button:hover {
        transform: scale(1.08);
        box-shadow: 0 12px 30px rgba(126, 34, 206, 0.5);
    }
    
    .result-card {
        padding: 50px;
        border-radius: 25px;
        text-align: center;
        margin: 30px 0;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    }
    
    .section-header {
        color: white;
        font-size: 2.2em;
        font-weight: 700;
        text-align: center;
        margin: 40px 0 30px 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }

    .agent-hero-band {
        margin: 0 auto 18px auto;
        max-width: 1080px;
        padding: 20px 24px;
        border-radius: 18px;
        border: 1px solid rgba(251, 191, 36, 0.45);
        background: linear-gradient(135deg, rgba(23, 14, 5, 0.48), rgba(50, 23, 6, 0.28));
        box-shadow: 0 16px 38px rgba(13, 13, 32, 0.24);
    }

    .agent-hero-title {
        margin: 0;
        color: #fef3c7;
        font-size: 1.25em;
        font-weight: 800;
        letter-spacing: 0.2px;
        text-align: center;
    }

    .agent-hero-subtitle {
        margin: 8px 0 16px 0;
        color: #fde68a;
        font-size: 1em;
        line-height: 1.65;
        text-align: center;
    }

    .agent-pill-row {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-wrap: wrap;
        gap: 10px;
    }

    .agent-pill {
        display: inline-flex;
        align-items: center;
        padding: 8px 12px;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.14);
        border: 1px solid rgba(255, 255, 255, 0.22);
        color: #f9fafb;
        font-size: 0.88em;
        font-weight: 700;
        letter-spacing: 0.2px;
    }

    .disclaimer-panel {
        background: linear-gradient(135deg, rgba(45, 25, 8, 0.82), rgba(88, 49, 12, 0.74));
        border: 2px solid #f59e0b;
        border-radius: 16px;
        padding: 22px 26px;
        margin-top: 20px;
        box-shadow: 0 14px 34px rgba(13, 13, 32, 0.28);
    }

    .disclaimer-title {
        margin: 0 0 10px 0;
        color: #fef3c7;
        font-size: 1.08em;
        font-weight: 900;
        letter-spacing: 0.2px;
    }

    .disclaimer-body {
        margin: 0;
        color: #fff7ed;
        font-size: 1.04em;
        line-height: 1.85;
        font-weight: 550;
    }

    @media (max-width: 768px) {
        .main-title {
            font-size: 2.55em;
            line-height: 1.1;
        }

        .subtitle {
            font-size: 1.06em;
            margin-bottom: 26px;
        }

        .card {
            padding: 24px;
            margin: 18px 0;
        }

        .section-header {
            font-size: 1.55em;
            margin: 24px 0 16px 0;
        }

        .agent-hero-band {
            padding: 16px;
            margin-bottom: 14px;
        }

        .agent-hero-title {
            font-size: 1.02em;
        }

        .agent-hero-subtitle {
            font-size: 0.94em;
        }

        .agent-pill {
            font-size: 0.82em;
            padding: 7px 10px;
        }

        .disclaimer-panel {
            padding: 16px 18px;
        }

        .disclaimer-title {
            font-size: 1em;
        }

        .disclaimer-body {
            font-size: 0.98em;
            line-height: 1.75;
        }
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_resources():
    models = load_all_models()
    columns = load_columns()
    scaler = load_scaler()
    return models, columns, scaler


models, columns, scaler = load_resources()


@st.cache_resource
def load_evaluation_data():
    """Load pre-computed test split for evaluation metrics and confusion matrix."""
    with open("models/test_data.pkl", "rb") as f:
        test_data = pickle.load(f)
    with open("models/metrics.json", "r") as f:
        metrics = json.load(f)
    return test_data, metrics


test_data, all_metrics = load_evaluation_data()


if "page" not in st.session_state:
    st.session_state.page = "intro"
if "user_data" not in st.session_state:
    st.session_state.user_data = None
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None
if "churn_prob_for_agent" not in st.session_state:
    st.session_state.churn_prob_for_agent = None

# ==================== PAGE 1: INTRO ====================
if st.session_state.page == "intro":
    st.markdown(
        "<h1 class='main-title'>🚀 Customer Churn Prediction AI</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p class='subtitle'>Predict customer churn with advanced machine learning algorithms</p>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown(
            """
        <div class='card'>
            <h2 style='color: #7e22ce; text-align: center; margin-bottom: 25px;'>Welcome to the Future of Customer Analytics</h2>
            <p style='font-size: 1.2em; line-height: 2; text-align: justify; color: #374151;'>
                Customer churn is one of the most critical challenges facing businesses today. 
                Our AI-powered platform leverages state-of-the-art machine learning algorithms to 
                predict which customers are likely to leave, enabling proactive retention strategies.
            </p>
            <p style='font-size: 1.2em; line-height: 2; text-align: justify; color: #374151; margin-top: 20px;'>
                With three powerful models - <strong style='color: #7e22ce;'>Logistic Regression</strong>, 
                <strong style='color: #7e22ce;'>Decision Tree</strong>, and <strong style='color: #7e22ce;'>Random Forest</strong> 
                - analyze customer behavior and make data-driven decisions.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
        <div class='agent-hero-band'>
            <p class='agent-hero-title'>Now with Agentic Retention Intelligence</p>
            <p class='agent-hero-subtitle'>
                Beyond churn prediction, the system now runs a multi-step LangGraph pipeline
                with RAG retrieval over domain research to generate grounded, personalized
                retention strategies with source-backed recommendations.
            </p>
            <div class='agent-pill-row'>
                <span class='agent-pill'>LangGraph: 4-node reasoning flow</span>
                <span class='agent-pill'>RAG: Chroma + 7 research PDFs</span>
                <span class='agent-pill'>LLM: Groq-powered intervention planning</span>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown(
        "<h2 class='section-header'>✨ Key Features</h2>", unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
        <div class='card'>
            <div style='font-size: 4em; margin-bottom: 20px;'>🎯</div>
            <h3 style='color: #7e22ce; margin-bottom: 15px;'>Multiple Models</h3>
            <p style='font-size: 1.1em; color: #6b7280; line-height: 1.8;'>
                Choose from three powerful ML algorithms with unique strengths.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div class='card'>
            <div style='font-size: 4em; margin-bottom: 20px;'>📊</div>
            <h3 style='color: #7e22ce; margin-bottom: 15px;'>Accurate Predictions</h3>
            <p style='font-size: 1.1em; color: #6b7280; line-height: 1.8;'>
                Get instant churn predictions with probability scores.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
        <div class='card'>
            <div style='font-size: 4em; margin-bottom: 20px;'>⚡</div>
            <h3 style='color: #7e22ce; margin-bottom: 15px;'>Real-time Analysis</h3>
            <p style='font-size: 1.1em; color: #6b7280; line-height: 1.8;'>
                Instant results with detailed probability analysis.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🚀 Start Prediction", use_container_width=True, key="start_btn"):
            st.session_state.page = "prediction"
            st.rerun()

# ==================== PAGE 2: PREDICTION INPUT ====================
elif st.session_state.page == "prediction":
    if st.button("🏠 Back to Home", use_container_width=True):
        st.session_state.page = "intro"
        st.session_state.user_data = None
        st.session_state.selected_model = None
        st.rerun()
    st.markdown(
        "<h1 class='main-title'>📋 Enter Customer Information</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p class='subtitle'>Fill in the details below to predict churn probability</p>",
        unsafe_allow_html=True,
    )

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 👤 Customer Demographics")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Has Partner", ["No", "Yes"])
            dependents = st.selectbox("Has Dependents", ["No", "Yes"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("### 📞 Phone Services")
            phone_service = st.selectbox("Phone Service", ["No", "Yes"])
            multiple_lines = st.selectbox(
                "Multiple Lines", ["No", "Yes", "No phone service"]
            )
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("### 🌐 Internet Services")
            internet_service = st.selectbox(
                "Internet Service", ["DSL", "Fiber optic", "No"]
            )
            online_security = st.selectbox(
                "Online Security", ["No", "Yes", "No internet service"]
            )
            online_backup = st.selectbox(
                "Online Backup", ["No", "Yes", "No internet service"]
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("### 🛡️ Additional Services")
            device_protection = st.selectbox(
                "Device Protection", ["No", "Yes", "No internet service"]
            )
            tech_support = st.selectbox(
                "Tech Support", ["No", "Yes", "No internet service"]
            )
            streaming_tv = st.selectbox(
                "Streaming TV", ["No", "Yes", "No internet service"]
            )
            streaming_movies = st.selectbox(
                "Streaming Movies", ["No", "Yes", "No internet service"]
            )
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("### 💳 Billing Information")
            contract = st.selectbox(
                "Contract Type", ["Month-to-month", "One year", "Two year"]
            )
            paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
            monthly_charges = st.number_input(
                "Monthly Charges ($)", 0.0, 200.0, 50.0, step=5.0
            )
            total_charges = st.number_input(
                "Total Charges ($)", 0.0, 10000.0, 500.0, step=50.0
            )
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            submitted = st.form_submit_button(
                "🔮 Predict Churn", use_container_width=True
            )

        if submitted:
            st.session_state.user_data = {
                "gender": gender,
                "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless_billing,
                "PaymentMethod": payment_method,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
            }
            st.session_state.page = "model_selection"
            st.rerun()

# ==================== PAGE 3: MODEL SELECTION ====================
elif st.session_state.page == "model_selection":
    if st.button("🏠 Back to prediction", use_container_width=True):
        st.session_state.page = "prediction"
        st.session_state.user_data = None
        st.session_state.selected_model = None
        st.rerun()
    st.markdown(
        "<h1 class='main-title'>🤖 Select Your Model</h1>", unsafe_allow_html=True
    )
    st.markdown(
        "<p class='subtitle'>Choose a machine learning model to analyze the customer data</p>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
        <div class='model-card'>
            <div style='font-size: 5em; margin-bottom: 20px;'>📊</div>
            <h2 style='color: #7e22ce; margin-bottom: 15px;'>Logistic Regression</h2>
            <p style='font-size: 1.1em; color: #6b7280; line-height: 1.8; margin-bottom: 20px;'>
                Linear model ideal for interpretability and baseline performance.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )
        if st.button(
            "Select Logistic Regression", use_container_width=True, key="lr_btn"
        ):
            st.session_state.selected_model = "Logistic Regression"
            st.session_state.page = "result"
            st.rerun()

    with col2:
        st.markdown(
            """
        <div class='model-card'>
            <div style='font-size: 5em; margin-bottom: 20px;'>🌳</div>
            <h2 style='color: #7e22ce; margin-bottom: 15px;'>Decision Tree</h2>
            <p style='font-size: 1.1em; color: #6b7280; line-height: 1.8; margin-bottom: 20px;'>
                Tree-based model with clear decision rules and interactions.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )
        if st.button("Select Decision Tree", use_container_width=True, key="dt_btn"):
            st.session_state.selected_model = "Decision Tree"
            st.session_state.page = "result"
            st.rerun()

    with col3:
        st.markdown(
            """
        <div class='model-card'>
            <div style='font-size: 5em; margin-bottom: 20px;'>🌲</div>
            <h2 style='color: #7e22ce; margin-bottom: 15px;'>Random Forest</h2>
            <p style='font-size: 1.1em; color: #6b7280; line-height: 1.8; margin-bottom: 20px;'>
                Ensemble method for superior accuracy and robustness.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )
        if st.button("Select Random Forest", use_container_width=True, key="rf_btn"):
            st.session_state.selected_model = "Random Forest"
            st.session_state.page = "result"
            st.rerun()

# ==================== PAGE 4: RESULT ====================
elif st.session_state.page == "result":
    model_name = st.session_state.selected_model
    model = models[model_name]

    model_icons = {
        "Logistic Regression": "📊",
        "Decision Tree": "🌳",
        "Random Forest": "🌲",
    }

    st.markdown(
        f"<h1 class='main-title'>{model_icons[model_name]} {model_name}</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p class='subtitle'>Churn Prediction Results</p>", unsafe_allow_html=True
    )

    processed = preprocess_input(st.session_state.user_data)
    prediction = model.predict(processed)[0]
    prob = model.predict_proba(processed)[0]

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if prediction == 1:
            st.markdown(
                f"""
            <div class='result-card' style='background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); color: white;'>
                <div style='font-size: 5em; margin-bottom: 20px;'>⚠️</div>
                <h1 style='font-size: 3em; margin-bottom: 20px;'>HIGH CHURN RISK</h1>
                <div style='font-size: 6em; font-weight: 900; margin: 30px 0;'>{prob[1]:.1%}</div>
                <p style='font-size: 1.4em; margin-top: 20px;'>
                    This customer is likely to churn. Consider retention strategies!
                </p>
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
            <div class='result-card' style='background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white;'>
                <div style='font-size: 5em; margin-bottom: 20px;'>✅</div>
                <h1 style='font-size: 3em; margin-bottom: 20px;'>LOW CHURN RISK</h1>
                <div style='font-size: 6em; font-weight: 900; margin: 30px 0;'>{prob[0]:.1%}</div>
                <p style='font-size: 1.4em; margin-top: 20px;'>
                    This customer is likely to stay. Keep up the excellent service!
                </p>
            </div>
            """,
                unsafe_allow_html=True,
            )

    st.markdown(
        "<h2 class='section-header'>📊 Churn Probability Gauge</h2>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=prob[1] * 100,
                domain={"x": [0, 1], "y": [0, 1]},
                title={
                    "text": "Churn Risk Level",
                    "font": {"size": 24, "color": "white"},
                },
                delta={
                    "reference": 50,
                    "increasing": {"color": "#ef4444"},
                    "decreasing": {"color": "#10b981"},
                },
                gauge={
                    "axis": {
                        "range": [None, 100],
                        "tickwidth": 2,
                        "tickcolor": "white",
                    },
                    "bar": {"color": "#7e22ce"},
                    "bgcolor": "white",
                    "borderwidth": 2,
                    "bordercolor": "white",
                    "steps": [
                        {"range": [0, 30], "color": "#10b981"},
                        {"range": [30, 70], "color": "#f59e0b"},
                        {"range": [70, 100], "color": "#ef4444"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 4},
                        "thickness": 0.75,
                        "value": prob[1] * 100,
                    },
                },
            )
        )

        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "white", "family": "Arial"},
            height=400,
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown(
        "<h2 class='section-header'>📈 Probability Breakdown</h2>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""
        <div class='card' style='background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; text-align: center;'>
            <h2 style='font-size: 2.5em; margin-bottom: 15px;'>Will Stay</h2>
            <div style='font-size: 4em; font-weight: 900;'>{prob[0]:.1%}</div>
            <p style='font-size: 1.2em; margin-top: 15px;'>Retention Probability</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
        <div class='card' style='background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); color: white; text-align: center;'>
            <h2 style='font-size: 2.5em; margin-bottom: 15px;'>Will Churn</h2>
            <div style='font-size: 4em; font-weight: 900;'>{prob[1]:.1%}</div>
            <p style='font-size: 1.2em; margin-top: 15px;'>Churn Probability</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown(
        "<h2 class='section-header'>🎯 Top Features Influencing This Prediction</h2>",
        unsafe_allow_html=True,
    )

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        importances = np.ones(len(columns))

    feature_df = (
        pd.DataFrame({"Feature": columns, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .head(10)
    )

    fig_features = px.bar(
        feature_df,
        y="Feature",
        x="Importance",
        orientation="h",
        title="Top 10 Most Important Features",
        color="Importance",
        color_continuous_scale=["#7e22ce", "#ec4899", "#f59e0b"],
        labels={"Importance": "Importance Score", "Feature": "Customer Features"},
    )

    fig_features.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.9)",
        font={"color": "white", "family": "Arial", "size": 12},
        title_font={"size": 20, "color": "white"},
        xaxis={"gridcolor": "rgba(255,255,255,0.3)"},
        yaxis={"gridcolor": "rgba(255,255,255,0.3)"},
        height=500,
        showlegend=False,
    )

    st.plotly_chart(fig_features, use_container_width=True)

    st.markdown(
        "<h2 class='section-header'>📊 Probability Distribution</h2>",
        unsafe_allow_html=True,
    )

    fig_prob = go.Figure(
        data=[
            go.Bar(
                x=["Will Stay", "Will Churn"],
                y=[prob[0] * 100, prob[1] * 100],
                marker=dict(
                    color=["#10b981", "#ef4444"], line=dict(color="white", width=2)
                ),
                text=[f"{prob[0]:.1%}", f"{prob[1]:.1%}"],
                textposition="outside",
                textfont=dict(size=20, color="white", family="Arial Black"),
            )
        ]
    )

    fig_prob.update_layout(
        title="Prediction Confidence",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.1)",
        font={"color": "white", "family": "Arial"},
        yaxis={
            "title": "Probability (%)",
            "gridcolor": "rgba(255,255,255,0.2)",
            "range": [0, 100],
        },
        xaxis={"title": "", "showgrid": False},
        title_font={"size": 20, "color": "white"},
        height=400,
        showlegend=False,
    )

    st.plotly_chart(fig_prob, use_container_width=True)

    # ==================== MODEL PERFORMANCE METRICS ====================
    st.markdown(
        "<h2 class='section-header'>📈 Model Performance Metrics</h2>",
        unsafe_allow_html=True,
    )

    m = all_metrics.get(model_name, {})

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    metric_style = (
        "background:white; border-radius:15px; padding:25px 10px; text-align:center;"
        "box-shadow:0 8px 25px rgba(0,0,0,0.15);"
    )

    with col_m1:
        st.markdown(
            f"<div style='{metric_style}'>"
            f"<div style='font-size:0.9em;color:#6b7280;font-weight:600;"
            f"text-transform:uppercase;letter-spacing:1px;'>Accuracy</div>"
            f"<div style='font-size:2.8em;font-weight:900;color:#7e22ce;margin:8px 0;'>"
            f"{m.get('accuracy', 0):.1%}</div></div>",
            unsafe_allow_html=True,
        )
    with col_m2:
        st.markdown(
            f"<div style='{metric_style}'>"
            f"<div style='font-size:0.9em;color:#6b7280;font-weight:600;"
            f"text-transform:uppercase;letter-spacing:1px;'>Precision</div>"
            f"<div style='font-size:2.8em;font-weight:900;color:#ec4899;margin:8px 0;'>"
            f"{m.get('precision', 0):.1%}</div></div>",
            unsafe_allow_html=True,
        )
    with col_m3:
        st.markdown(
            f"<div style='{metric_style}'>"
            f"<div style='font-size:0.9em;color:#6b7280;font-weight:600;"
            f"text-transform:uppercase;letter-spacing:1px;'>Recall</div>"
            f"<div style='font-size:2.8em;font-weight:900;color:#f59e0b;margin:8px 0;'>"
            f"{m.get('recall', 0):.1%}</div></div>",
            unsafe_allow_html=True,
        )
    with col_m4:
        st.markdown(
            f"<div style='{metric_style}'>"
            f"<div style='font-size:0.9em;color:#6b7280;font-weight:600;"
            f"text-transform:uppercase;letter-spacing:1px;'>F1 Score</div>"
            f"<div style='font-size:2.8em;font-weight:900;color:#10b981;margin:8px 0;'>"
            f"{m.get('f1', 0):.1%}</div></div>",
            unsafe_allow_html=True,
        )

    # Confusion Matrix
    st.markdown(
        "<h2 class='section-header'>🔢 Confusion Matrix</h2>",
        unsafe_allow_html=True,
    )

    from sklearn.metrics import confusion_matrix as sk_confusion_matrix
    import plotly.figure_factory as ff

    y_pred_eval = model.predict(test_data["X_test_scaled"])
    cm = sk_confusion_matrix(test_data["y_test"], y_pred_eval)

    cm_labels = ["Will Stay", "Will Churn"]
    fig_cm = ff.create_annotated_heatmap(
        z=cm,
        x=cm_labels,
        y=cm_labels,
        colorscale=[[0, "#f0f4ff"], [1, "#7e22ce"]],
        showscale=False,
        annotation_text=[[str(v) for v in row] for row in cm],
    )
    fig_cm.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white", "family": "Arial", "size": 14},
        height=350,
        xaxis={"title": "Predicted", "side": "bottom"},
        yaxis={"title": "Actual", "autorange": "reversed"},
        margin={"l": 80, "r": 40, "t": 40, "b": 60},
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.plotly_chart(fig_cm, use_container_width=True)

    # ---- AI RETENTION STRATEGY SECTION ----
    st.markdown(
        "<hr style='border:1px solid rgba(255,255,255,0.2); margin: 40px 0;'>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h2 class='section-header'>🤖 AI-Powered Retention Strategy</h2>",
        unsafe_allow_html=True,
    )

    col_ai1, col_ai2, col_ai3 = st.columns([1, 2, 1])
    with col_ai2:
        st.markdown(
            f"""
        <div class='card' style='text-align:center;'>
            <div style='font-size:3em; margin-bottom:15px;'>🧠</div>
            <h3 style='color:#7e22ce;'>LangGraph Retention Agent</h3>
            <p style='color:#6b7280; font-size:1.1em; line-height:1.8;'>
                Run the agentic AI pipeline to analyse this customer's churn risk,
                retrieve evidence-based strategies from the knowledge base, and
                generate a personalised retention plan.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )
        if st.button(
            "🧠 Generate AI Retention Strategy",
            use_container_width=True,
            key="agent_btn",
        ):
            st.session_state.churn_prob_for_agent = float(prob[1])
            st.session_state.page = "agent_report"
            st.rerun()

    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔄 Try Another Model", use_container_width=True):
            st.session_state.page = "model_selection"
            st.rerun()

    with col2:
        if st.button("📝 New Prediction", use_container_width=True):
            st.session_state.page = "prediction"
            st.session_state.user_data = None
            st.rerun()

    with col3:
        if st.button("🏠 Back to Home", use_container_width=True):
            st.session_state.page = "intro"
            st.session_state.user_data = None
            st.session_state.selected_model = None
            st.rerun()

# ==================== PAGE 5: AGENT RETENTION REPORT ====================
elif st.session_state.page == "agent_report":
    if st.button("⬅️ Back to Results", use_container_width=True, key="back_to_result"):
        st.session_state.page = "result"
        st.rerun()

    st.markdown(
        "<h1 class='main-title'>🤖 AI Retention Strategy Report</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p class='subtitle'>LangGraph Agentic Pipeline — Powered by Groq LLM + RAG</p>",
        unsafe_allow_html=True,
    )

    customer_data = st.session_state.user_data
    churn_prob = st.session_state.churn_prob_for_agent

    if customer_data is None or churn_prob is None:
        st.warning("No customer data found. Please start from the prediction page.")
        if st.button("Go to Prediction"):
            st.session_state.page = "prediction"
            st.rerun()
    else:
        if not AGENT_AVAILABLE or retention_agent is None:
            st.error(
                "The AI agent could not be loaded. Check your terminal for the import error."
            )
            st.stop()

        with st.spinner(
            "🔍 Assessing risk profile... 📚 Retrieving retention strategies... "
            "🧠 Planning intervention... 📝 Generating report..."
        ):
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

        report = result.get("retention_report", {})
        risk_level = result.get("risk_level", "unknown")
        risk_drivers = result.get("risk_drivers", [])

        # Risk level badge
        risk_colors = {
            "high": ("linear-gradient(135deg, #ef4444 0%, #dc2626 100%)", "⚠️"),
            "medium": ("linear-gradient(135deg, #f59e0b 0%, #d97706 100%)", "⚡"),
            "low": ("linear-gradient(135deg, #10b981 0%, #059669 100%)", "✅"),
        }
        bg, icon = risk_colors.get(
            risk_level, ("linear-gradient(135deg,#6b7280,#4b5563)", "❓")
        )

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(
                f"""
            <div class='result-card' style='background:{bg}; color:white;'>
                <div style='font-size:4em;'>{icon}</div>
                <h2 style='font-size:2.5em; margin:10px 0;'>{risk_level.upper()} CHURN RISK</h2>
                <div style='font-size:3.5em; font-weight:900;'>{churn_prob:.1%}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # Risk summary
        st.markdown(
            "<h2 class='section-header'>📋 Risk Summary</h2>", unsafe_allow_html=True
        )
        col1, col2, col3 = st.columns([1, 4, 1])
        with col2:
            st.markdown(
                f"""
            <div class='card'>
                <p style='font-size:1.15em; color:#374151; line-height:2;'>
                    {report.get("risk_summary", "Risk analysis completed.")}
                </p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # Risk drivers
        if risk_drivers:
            st.markdown(
                "<h2 class='section-header'>🔍 Key Risk Drivers</h2>",
                unsafe_allow_html=True,
            )
            col1, col2, col3 = st.columns([1, 4, 1])
            with col2:
                drivers_html = "".join(
                    f"<li style='padding:6px 0; font-size:1.05em; color:#374151;'>• {d}</li>"
                    for d in risk_drivers
                )
                st.markdown(
                    f"""
                <div class='card'>
                    <ul style='list-style:none; padding:0; margin:0;'>{drivers_html}</ul>
                </div>
                """,
                    unsafe_allow_html=True,
                )

        # Recommended actions
        actions = report.get("recommended_actions", [])
        if actions:
            st.markdown(
                "<h2 class='section-header'>💡 Recommended Retention Actions</h2>",
                unsafe_allow_html=True,
            )
            for i, action in enumerate(actions, 1):
                col1, col2, col3 = st.columns([1, 4, 1])
                with col2:
                    st.markdown(
                        f"""
                    <div class='card' style='margin:10px 0; padding:25px 35px;
                         border-left: 5px solid #7e22ce;'>
                        <span style='font-size:1.4em; font-weight:700;
                              color:#7e22ce; margin-right:12px;'>#{i}</span>
                        <span style='font-size:1.1em; color:#374151;'>{action}</span>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

        # Agent reasoning
        reasoning = report.get("reasoning", "")
        if reasoning:
            st.markdown(
                "<h2 class='section-header'>🧠 Agent Reasoning</h2>",
                unsafe_allow_html=True,
            )
            col1, col2, col3 = st.columns([1, 4, 1])
            with col2:
                st.markdown(
                    f"""
                <div class='card' style='background:linear-gradient(135deg,#f0f4ff,#faf5ff);'>
                    <p style='font-size:1.05em; color:#4b5563; line-height:1.9;
                              font-style:italic;'>{reasoning}</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

        # Knowledge base sources
        sources = report.get("sources", [])
        if sources:
            st.markdown(
                "<h2 class='section-header'>📚 Knowledge Base Sources</h2>",
                unsafe_allow_html=True,
            )
            col1, col2, col3 = st.columns([1, 4, 1])
            with col2:
                sources_html = "".join(
                    f"<li style='padding:5px 0; font-size:0.95em; color:#6b7280;'>{s}</li>"
                    for s in sources
                )
                st.markdown(
                    f"""
                <div class='card'>
                    <ul style='list-style:none; padding:0; margin:0;'>{sources_html}</ul>
                </div>
                """,
                    unsafe_allow_html=True,
                )

        # Ethical disclaimer - always shown
        disclaimer = report.get("ethical_disclaimer", "")
        if disclaimer:
            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 4, 1])
            with col2:
                st.markdown(
                    f"""
                <div class='disclaimer-panel'>
                    <p class='disclaimer-title'>⚠️ Ethical Disclaimer</p>
                    <p class='disclaimer-body'>
                        {disclaimer}
                    </p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

        # Show any agent error in a non-blocking way
        if result.get("error"):
            st.warning(f"Agent note: {result['error']}")

        # Navigation
        st.markdown("<br><br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button(
                "🔄 Try Another Model", use_container_width=True, key="agent_model"
            ):
                st.session_state.page = "model_selection"
                st.rerun()
        with col2:
            if st.button(
                "📝 New Prediction", use_container_width=True, key="agent_new"
            ):
                st.session_state.page = "prediction"
                st.session_state.user_data = None
                st.rerun()
        with col3:
            if st.button("🏠 Back to Home", use_container_width=True, key="agent_home"):
                st.session_state.page = "intro"
                st.session_state.user_data = None
                st.session_state.selected_model = None
                st.rerun()
