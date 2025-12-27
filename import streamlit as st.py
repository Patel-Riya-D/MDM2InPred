import streamlit as st
import pandas as pd
import joblib
import tempfile
import shutil
import os
import math
from padelpy import from_smiles
from rdkit import Chem
import base64
import cohere

# ------------------------------
# CONFIG - Update paths
# ------------------------------
MODEL_PATHS = {
    "Model 1": "lightgbm.pkl",
    "Model 2": "rf.pkl"
}
FEATURE_PATHS = {
    "Model 1": "newm1_aligned_376.csv",
    "Model 2": "rf_ready_newm2.csv"
}
DATASET_PATH = "main.csv"

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(page_title="MDM2 pIC50 Prediction", layout="wide")

# ------------------------------
# Custom CSS (Theme + Chatbot)
# ------------------------------
st.markdown(
    """
<style>
/* Global font & background */
.stApp {
    background: #ffffff;
    color: #1a1a1a;
}

/* Title */
.main-title {
    text-align: center;
    font-size: 40px;
    font-family: 'Times New Roman', serif; 
    font-weight: bold;
    margin-bottom: 20px;
    color: #006666;
}

/* Tabs */
.stTabs [role="tablist"] {
    gap: 0px;
    display: flex;
    justify-content: center;
    border-bottom: 2px solid #006666;
    overflow-x: auto;
}

.stTabs [role="tab"] span {
    font-size: 18px !important;
    font-weight: 600 !important;
    color: #006666 !important;
    font-family: 'Segoe UI', sans-serif !important;
}

.stTabs [role="tab"] {
    padding: 8px 16px !important;
    background-color: #e6ffff !important;
    border-radius: 0px 0px 0 0 !important;
}

.stTabs [aria-selected="true"] {
    background-color: #006666 !important;
    color: white !important;
    border-radius: 10px 10px 0 0;
    box-shadow: 0px -4px 12px rgba(0,0,0,0.25);
    transform: translateY(-4px) scale(1.05);
    transition: all 0.3s ease-in-out;
}

/* Horizontal line */
hr {
    border: none;
    border-top: 2px solid #006666;
    margin: 10px auto;
    width: 95%;
    max-width: 1200px;
}

/* Buttons */
.stButton>button {
    background-color: #006666;
    color: white;
    border-radius: 8px;
    font-size: 16px;
    font-weight: bold;
    padding: 6px 20px;
}
.stButton>button:hover {
    background-color: #002244;
}

/* Small example text */
.example-text {
    font-size: 20px;
    color: #444;
    font-style: italic;
}

/* Floating Chat Icon */
#floating-chat-icon {
    position: fixed;
    bottom: 25px;
    right: 25px;
    width: 60px;
    height: 60px;
    background-color: #006666;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    cursor: pointer;
    z-index: 999999;
    border: 3px solid white;
}
#floating-chat-icon:hover {
    background-color: #004d4d;
    transform: scale(1.08);
}

/* Chat Window */
.chat-window {
    position: fixed;
    bottom: 90px;
    right: 25px;
    width: 350px;
    height: 500px;
    background: white;
    border-radius: 14px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.3);
    border: 2px solid #006666;
    display: flex;
    flex-direction: column;
    z-index: 999998;
    font-family: 'Segoe UI', sans-serif;
}

/* Header */
.chat-header {
    background: #006666;
    color: white;
    padding: 12px 15px;
    border-radius: 12px 12px 0 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.chat-title {
    font-size: 15px;
    font-weight: bold;
}
.chat-close {
    background: none;
    border: none;
    color: white;
    font-size: 18px;
    cursor: pointer;
    padding: 0;
    width: 24px;
    height: 24px;
    border-radius: 50%;
}
.chat-close:hover {
    background: rgba(255,255,255,0.2);
}

/* Messages */
.chat-messages {
    flex: 1;
    padding: 10px;
    overflow-y: auto;
    background: #f8f9fa;
}
.message {
    padding: 9px 13px;
    border-radius: 16px;
    max-width: 80%;
    word-wrap: break-word;
    font-size: 14px;
    line-height: 1.4;
    margin: 5px 0;
}
.user-message {
    background: #006666;
    color: white;
    margin-left: auto;
    border-bottom-right-radius: 4px;
}
.bot-message {
    background: #e9ecef;
    color: #333;
    margin-right: auto;
    border-bottom-left-radius: 4px;
}
.welcome-message {
    text-align: center;
    color: #666;
    font-style: italic;
    padding: 15px;
    font-size: 14px;
    background: white;
    border-radius: 10px;
    margin: 10px;
}

/* Input area */
.chat-input-container {
    padding: 10px;
    border-top: 1px solid #dee2e6;
    background: white;
    border-radius: 0 0 12px 12px;
}
.chat-input-row {
    display: flex;
    gap: 8px;
    align-items: center;
}
.chat-input {
    flex: 1;
    padding: 9px 12px;
    border: 1px solid #ced4da;
    border-radius: 18px;
    outline: none;
    font-size: 14px;
}
.chat-input:focus {
    border-color: #006666;
}
.chat-send-btn {
    background: #006666;
    color: white;
    border: none;
    padding: 9px 15px;
    border-radius: 18px;
    cursor: pointer;
    font-size: 14px;
    font-weight: 600;
}
.chat-send-btn:hover {
    background: #004d4d;
}
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------
# Cohere Client & Chatbot State
# ------------------------------
co = cohere.Client(st.secrets["COHERE_API_KEY"])

if "chat_open" not in st.session_state:
    st.session_state.chat_open = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ------------------------------
# Converter Functions
# ------------------------------
def pIC50_to_IC50(pIC50):
    return 10 ** (-pIC50)

def IC50_to_pIC50(IC50):
    return -math.log10(IC50)

# ------------------------------
# Helper Functions
# ------------------------------
def filter_valid_smiles(smiles_list):
    valid, invalid = [], []
    for s in smiles_list:
        if Chem.MolFromSmiles(s):
            valid.append(s)
        else:
            invalid.append(s)
    return valid, invalid

def generate_descriptors_safe_individual(smiles_list):
    tmpdir = tempfile.mkdtemp()
    os.environ["JAVA_TOOL_OPTIONS"] = "-Xmx4G"
    all_desc = []
    skipped = []

    progress_bar = st.progress(0)

    for i, smi in enumerate(smiles_list):
        try:
            batch_file = os.path.join(tmpdir, f"desc_{i}.csv")
            from_smiles([smi], output_csv=batch_file, fingerprints=True, timeout=600)
            if os.path.exists(batch_file):
                desc_df = pd.read_csv(batch_file)
                all_desc.append(desc_df)
            else:
                skipped.append(smi)
        except Exception:
            skipped.append(smi)
        progress_bar.progress((i + 1) / len(smiles_list))

    shutil.rmtree(tmpdir, ignore_errors=True)

    if skipped:
        st.warning(f"{len(skipped)} SMILES could not be processed. Examples: {skipped[:5]}")

    return pd.concat(all_desc, ignore_index=True)

def run_prediction(model_key, smiles_input, uploaded):
    if smiles_input.strip():
        smiles_list = [line.strip() for line in smiles_input.strip().splitlines()]
    elif uploaded:
        smiles_list = [line.decode("utf-8").strip() for line in uploaded.readlines()]
    else:
        st.error("Please provide SMILES input or upload a file.")
        return

    smiles_list, invalid_smiles = filter_valid_smiles(smiles_list)
    if invalid_smiles:
        st.warning(f"{len(invalid_smiles)} invalid SMILES removed. Examples: {invalid_smiles[:5]}")
    if not smiles_list:
        st.error("No valid SMILES to process!")
        return

    with st.spinner("Generating descriptors using PaDELPy..."):
        desc_df = generate_descriptors_safe_individual(smiles_list)

    model = joblib.load(MODEL_PATHS[model_key])
    training_features = pd.read_csv(FEATURE_PATHS[model_key]).columns.tolist()

    common_features = [f for f in training_features if f in desc_df.columns]
    X = desc_df[common_features]
    if X.empty:
        st.error("No matching features found between descriptors and training data!")
        return

    with st.spinner(f"Predicting activity using {model_key}..."):
        prediction = model.predict(X)

    activity = ["Likely Inhibitor" if p >= 6 else "Likely Non-inhibitor" for p in prediction]
    ic50_values = [pIC50_to_IC50(p) for p in prediction]

    st.markdown(
        """
        <h2 style='color:#006666; font-size:40px; font-family:"Times New Roman", serif; font-weight:bold;'>
            Prediction Results
        </h2>
        """,
        unsafe_allow_html=True,
    )

    results_df = pd.DataFrame(
        {
            "SMILES": desc_df["Name"],
            "pIC₅₀ value": prediction,
            "Prediction": activity,
        }
    )

    display_df = results_df.copy()
    display_df["pIC₅₀ value"] = display_df["pIC₅₀ value"].map(lambda x: f"{x:.4f}")

    def style_table(row):
        base_color = "#f9ffff" if row.name % 2 == 0 else "#ffffff"
        styles = [
            f"background-color: {base_color}; color: #00332e; font-size:16px; text-align:center;"
        ] * len(row)

        if row["Prediction"] == "Likely Inhibitor":
            styles[-1] = "background-color: #006666; color: white; font-weight: bold; text-align:center;"
        elif row["Prediction"] == "Likely Non-inhibitor":
            styles[-1] = "background-color: #cce6ff; color: #065f46; font-weight: bold; text-align:center;"

        return styles

    display_df = display_df.reset_index(drop=True)
    styled_df = display_df.style.apply(style_table, axis=1)

    styled_df = styled_df.set_table_styles(
        [
            {
                "selector": "th",
                "props": [
                    ("background-color", "#006666"),
                    ("color", "white"),
                    ("font-weight", "bold"),
                    ("text-align", "center"),
                    ("font-size", "18px"),
                    ("padding", "12px"),
                ],
            },
            {
                "selector": "td",
                "props": [
                    ("padding", "12px"),
                    ("text-align", "center"),
                    ("border", "1px solid #ddd"),
                ],
            },
        ]
    )

    table_html = styled_df.to_html(index=False)
    full_html = '<div style="width:100%; overflow-x:auto;">' + table_html + "</div>"

    st.markdown(full_html, unsafe_allow_html=True)

    csv_data = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Prediction Results (CSV)",
        data=csv_data,
        file_name=f"{model_key}_predictions.csv",
        mime="text/csv",
    )

# ------------------------------
# Chatbot Context & Logic
# ------------------------------
APP_CONTEXT = """
This is the MDM2InPred dashboard.

Modules:

1) Home:
   - Describes the biological background of MDM2, its interaction with p53, and its role in cancer.
   - Explains the importance of small-molecule MDM2 inhibitors.

2) Prediction:
   - Predicts the pIC50 value of user-provided molecules and classifies them as MDM2 inhibitors or non-inhibitors.
   - User can either paste SMILES strings or upload a .smi file (up to 200 MB).
   - Two machine learning models are available: LightGBM and Random Forest.
   - Models are trained on molecular descriptors generated using PaDEL.

3) Converter:
   - Performs bidirectional conversion between IC50 (in M) and pIC50 using:
     pIC50 = -log10(IC50).
   - Users can input either pIC50 or IC50 and get the converted value.

4) Dataset:
   - Provides access to the training, test, and external validation sets used to develop the models.
   - Separate datasets are available for the LightGBM and Random Forest models.

5) Help:
   - Provides basic instructions on how to use the dashboard and includes a video tutorial.

6) Contact:
   - Shows contact information and team profiles for the developers or researchers.
"""

SYSTEM_INSTRUCTIONS = """
You are an assistant for the MDM2InPred Streamlit dashboard.
Use the APP CONTEXT to answer questions about:
- How to use each module (Prediction, Converter, Dataset, Help, Contact)
- General concepts: MDM2, p53, IC50, pIC50, inhibitors, SMILES, machine learning models (LightGBM, Random Forest).

RULES:
- If the user asks for exact internal data (CSV contents, weights, hidden parameters, or precise training details not in the context), say that you do not have direct access and ask them to check the Dataset or Prediction tab.
- Do NOT invent experimental results, exact pIC50 values, or dataset entries.
- If you are unsure, clearly say you are not sure and guide the user to the appropriate tab in the dashboard.
- Be clear, concise, and user-friendly.
"""

def chatbot_reply(user_text: str) -> str:
    prompt = f"""
System:
{SYSTEM_INSTRUCTIONS}

APP CONTEXT:
{APP_CONTEXT}

User question:
{user_text}
"""
    try:
        response = co.chat(
            model="command-r-plus",  # Updated to a valid Cohere model
            message=prompt,
            temperature=0.2,
            max_tokens=300,
        )
        return response.text
    except co.exceptions.CohereError as e:  # Specific error handling
        return f"Cohere API error: {str(e)}. Please check your API key or try again."
    except Exception as e:
        return f"Unexpected error: {str(e)}. Please try again later."

# ------------------------------
# Chatbot Widget (Using Streamlit Components)
# ------------------------------
def render_chat_widget():
    # Floating chat icon as a Streamlit button
    if st.sidebar.button("💬 Open Chat", key="open_chat"):
        st.session_state.chat_open = True
    if st.session_state.chat_open:
        st.sidebar.button("❌ Close Chat", key="close_chat", on_click=lambda: setattr(st.session_state, 'chat_open', False))

    if not st.session_state.chat_open:
        return

    # Chat window in sidebar
    with st.sidebar:
        st.markdown("### 💬 MDM2 Assistant")
        chat_container = st.container(height=400)
        with chat_container:
            if not st.session_state.chat_history:
                st.markdown(
                    '<div class="welcome-message">Hello! Ask me anything about MDM2, pIC50, SMILES, Prediction module, IC50 converter, dataset etc.</div>',
                    unsafe_allow_html=True,
                )
            else:
                for msg in st.session_state.chat_history[-50:]:  # Limit to last 50 messages
                    if msg["role"] == "user":
                        st.markdown(f'<div class="message user-message">{msg["text"]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="message bot-message">{msg["text"]}</div>', unsafe_allow_html=True)

        # Input form using Streamlit
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_input("Type your message...", key="chat_input")
            submitted = st.form_submit_button("Send")
            if submitted and user_input.strip():
                st.session_state.chat_history.append({"role": "user", "text": user_input})
                with st.spinner("Thinking..."):
                    bot_response = chatbot_reply(user_input)
                st.session_state.chat_history.append({"role": "bot", "text": bot_response})
                st.rerun()  # Refresh to show new messages

# ------------------------------
# Main Title
# ------------------------------
st.markdown(
    "<div class='main-title'>MDM2InPred: Prediction of MDM2 Inhibitors</div>",
    unsafe_allow_html=True,
)

#