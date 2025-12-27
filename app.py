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
DATASET_PATH = "main.csv"  # your main dataset file

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(page_title="MDM2 pIC50 Prediction", layout="wide")

# ------------------------------
# Custom CSS (Theme)
# ------------------------------
st.markdown("""
<style>
/* Global font & background */
.stApp {
    background: #ffffff;
    color: #1a1a1a;
}

/* Title */
.main-title {
    text-align: center;
    font-size: 10px;
    font-family: 'zimula'; 
    font-weight: bold;
    margin-bottom: 0px;
}

/* Tabs */
/* ===== CLEAN TAB FIX (ONE ROW, CENTERED) ===== */

.stTabs [role="tablist"] {
    display: flex;
    justify-content: center;
    gap: 6px;
    border-bottom: 2px solid #006666;
    flex-wrap: nowrap;   /* 🔑 VERY IMPORTANT */
}

/* Individual tab */
.stTabs [role="tab"] {
    padding: 30px 28px !important;   /* ✅ reduced */
    background-color: #e6ffff !important;
    border-radius: 10px 10px 0 0 !important;
}

/* Tab text */
.stTabs [role="tab"] span {
    font-size: 18px !important;
    font-weight: 600 !important;
    color: #006666 !important;
    font-family: "Times New Roman", serif !important;
}

/* Active tab */
.stTabs [aria-selected="true"] {
    background-color: #006666 !important;
    color: white !important;
    box-shadow: 0px -3px 8px rgba(0,0,0,0.2);
    transform: none !important;   /* ❌ no lift */
}

/* Buttons */
.stButton>button {
    background-color: #006666;
    color: white;
    border-radius: 8px;
    font-size: 100px;
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
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Cohere Client & Chatbot State
# ------------------------------
# IMPORTANT: set COHERE_API_KEY in .streamlit/secrets.toml
# COHERE_API_KEY = "your-key-here"
co = cohere.Client(st.secrets["COHERE_API_KEY"])

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"role": "user"/"bot", "text": "..."}

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
    MAX_SMILES = 30
    if len(smiles_list) > MAX_SMILES:
        st.error("Maximum 30 SMILES allowed on Streamlit Cloud.")
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

    st.markdown("<div class='result-box'>", unsafe_allow_html=True)
    st.markdown(
        """
        <h2 style='color:#006666; font-size:40px; font-family:"Times New Roman", serif; font-weight:bold;'>
            Prediction Results
        </h2>
        """,
        unsafe_allow_html=True
    )

    # numeric results for CSV
    results_df = pd.DataFrame({
        "SMILES": desc_df["Name"],
        "pIC₅₀ value": prediction,
        "Prediction": activity
    })

    # display copy with pIC₅₀ converted to string → left alignment
    display_df = results_df.copy()
    display_df["pIC₅₀ value"] = display_df["pIC₅₀ value"].map(lambda x: f"{x:.4f}")

    # 🎨 Apply color styling based on Prediction column
    def style_table(row):
        base_color = '#f9ffff' if row.name % 2 == 0 else '#ffffff'
        styles = [f'background-color: {base_color}; color: #00332e; font-size:16px; text-align:center;'] * len(row)

        # Highlight Prediction column
        if row["Prediction"] == "Likely Inhibitor":
            styles[-1] = 'background-color: #006666; color: white; font-weight: bold; text-align:center;'
        elif row["Prediction"] == "Likely Non-inhibitor":
            styles[-1] = 'background-color: #cce6ff; color: #065f46; font-weight: bold; text-align:center;'

        return styles

    display_df = display_df.reset_index(drop=True)
    styled_df = display_df.style.apply(style_table, axis=1)

    styled_df = styled_df.set_table_styles(
        [
            {
                'selector': 'th',
                'props': [
                    ('background-color', '#006666'),
                    ('color', 'white'),
                    ('font-weight', 'bold'),
                    ('text-align', 'center'),
                    ('font-size', '18px'),
                    ('padding', '12px')
                ]
            },
            {
                'selector': 'td',
                'props': [
                    ('padding', '12px'),
                    ('text-align', 'center'),
                    ('border', '1px solid #ddd')
                ]
            }
        ]
    )

    # Inject global CSS for table
    st.markdown("""
    <style>
        table {
            width: 100% !important;
            border-collapse: collapse;
            border-radius: 10px;
            overflow: hidden;
        }
        table th {
            text-align: center !important;
        }
        table tr:hover {
            background-color: #e6ffff !important;
        }
    </style>
    """, unsafe_allow_html=True)

    table_html = styled_df.to_html(index=False)
    full_html = '<div style="width:100%; overflow-x:auto;">' + table_html + '</div>'

    # Render
    st.markdown(full_html, unsafe_allow_html=True)

    # keep numeric values in CSV
    csv_data = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Prediction Results (CSV)",
        data=csv_data,
        file_name=f"{model_key}_predictions.csv",
        mime="text/csv"
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
    """
    Call Cohere with context-enhanced prompt.
    """
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
            model="command-a-03-2025",   # ✅ NEW, currently live model
            message=prompt,
            temperature=0.2,
            max_tokens=300
        )
        return response.text
    except Exception as e:
        # Keep error short for users
        return "Sorry, I could not generate an answer right now. Please try again later."
 
# ------------------------------
# Main Title
# ------------------------------
st.markdown("<div class='main-title'>MDM2InPred: Prediction of MDM2 Inhibitors</div>", unsafe_allow_html=True)

# ------------------------------
# Tabs
# ------------------------------
# Custom CSS for tabs font size
st.markdown("""
<style>
/* Increase tab font size */
.stTabs [role="tab"] p {
    font-size: 20px !important;   /* adjust size */
    font-family: times new roman;
    padding-bottom: 0px !important;
}
</style>
""", unsafe_allow_html=True)

# Tabs (added Chatbot)
tab_home, tab_pred, tab_con, tab_data, tab_help, tab_contact, tab_chat = st.tabs(
    ["Home", "Prediction", "Converter", "Dataset", "Help", "Contact", "Ask AI"]
)

# ------------------------------
# HOME
# ------------------------------
with tab_home:
    col1, col2 = st.columns([1.3, 2])

    # Left side (Image)
    with col1:
        st.markdown(
            """
            <style>
            .custom-img img {
                height: 5000px;   /* make image taller */
                border-radius: 12px;
                box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
                margin-left: 200px;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        st.markdown('<div class="custom-img">', unsafe_allow_html=True)
        st.image("img1.jpg", use_container_width=True, output_format="PNG")
        st.markdown('</div>', unsafe_allow_html=True)

    # Right side (Intro Paragraph)
    with col2:
        st.markdown(
            """
            <style>
            .intro-box {
                font-size: 18px;
                line-height: 1.6;
                text-align: justify;
                margin-top: 90px;
                margin-left: 100px;
                font-family:book antiqua;
            }
            </style>
            <div class="intro-box">
                Murine double minute 2 (MDM2) is a p53-specific E3 ubiquitin ligase that regulates 
                the cell cycle, DNA repair, apoptosis, and oncogene activation through both p53-dependent 
                and independent pathways. MDM2 has emerged as the primary cellular antagonist of p53. 
                Interestingly, MDM2 is itself a product of a p53-inducible gene, and the two are connected 
                through an autoregulatory negative feedback loop that keeps p53 levels low in unstressed cells.  
                MDM2 binds directly to the N-terminal transactivation domain of p53, blocking its transcriptional 
                activity, which leads to nuclear export of p53, followed by ubiquitination and directing it to 
                the 26S proteasome for subsequent proteasomal degradation. Under oncogenic stress, ARF sequesters 
                MDM2 in the nucleolus, preventing p53 degradation and enabling the transcription of genes such as 
                p21 (cell cycle arrest), BAX, and PUMA (apoptosis).  
                MDM2 overexpression is observed in multiple cancers, underscoring its carcinogenic potential 
                and therapeutic relevance. In cancer pharmacology, small-molecule inhibitors are designed to 
                prevent the complex formation between MDM2 and p53 by blocking MDM2’s binding site, thereby 
                restoring p53’s function.
            </div>
            """,
            unsafe_allow_html=True
        )

    # Module section styling
    st.markdown("""
    <style>
    .module-box1 {
        background: #ffffff;
        height: 200px; 
        width: 400px;
        padding: 20px;
        margin-top: 25px;
        margin-left: 150px;
        font-family:book antiqua;
    }
    .module-box2 {
        background: #ffffff;
        height: 200px; 
        width: 400px;
        padding: 20px;
        margin-top: -200px;
        margin-left: 700px;
        font-family:book antiqua;
    }
    .module-title {
        font-size: 24px;
        font-weight: bold;
        color: #006666;
        margin-bottom: 10px;
        text-align: center;
    }
    .module-text {
        font-size: 16px;
        color: #333;
        margin-bottom: 15px;
        text-align: justify;
    }
    .module {
        background: #ffffff;
        height: 60px; 
        width: 1270px;    
        margin-top: -10px;
    }
    .main-title {
        font-size: 40px;
        font-weight: bold;
        color: #006666;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="module">
            <div class="main-title">Modules</div>
        </div>
        """, unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="module-box1">
            <div class="module-title">Prediction</div>
            <div class="module-text">
                It enables users to predict the pIC50 value of query
                and whether it is an inhibitor or non-inhibitor of MDM2.
                The user can either paste SMILES strings directly or upload a .smi file upto 200MB in size.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="module-box2">
            <div class="module-title">Converter</div>
            <div class="module-text">
                It enables bidirectional conversion between IC50 (in M)
                and pIC50. Users can convert the predicted output obtained 
                in pIC50 to IC50 using this module.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ------------------------------
# PREDICTION
# ------------------------------
with tab_pred:
    st.markdown("""
    <style>
    .prediction-box {
        background-color: #e6ffff;
        height: 180px;
        width: 1270px;
        padding: 0px;
        margin-bottom: 30px;
        margin-top: 15px;
    }
    .prediction-text {
        font-size: 18px;
        text-align: justify;
        font-family:book antiqua;
    }
    .prediction-title {
        font-size: 30px;
        text-align: center;
        color: #006666;
        font-weight: bold;
        font-family: zumila;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""     
        <div class="converter-box">
            <div class="converter-title">MDM2 Prediction Module</div>
            <div class="converter-text">
                This module predicts the pIC50 value of query molecules and also predicts whether they are
                inhibitors or non-inhibitors of MDM2. Light Gradient Boosting Machine (LightGBM) and 
                Random Forest (RF) machine learning algorithms have been implemented in the backend, and 
                users can select between both for prediction. The result will be visible in tabular format 
                and can also be downloaded as a CSV file. For more information, please refer to the Help page.
            </div>
        </div>
        """, unsafe_allow_html=True)


    st.markdown("""
    <style>
    /* Style for st.radio label (Select Model:) */
    div[data-testid="stRadio"] p {
        font-size: 16px !important;
        font-family: zumila;
        font-weight: normal !important;
        color: #000000 !important;
    }

    /* Style for st.write text like "You selected:" */
    div[data-testid="stMarkdownContainer"] p {
        font-size: 16px !important;
        font-family: zumila;
        font-weight: normal !important;
        color: #000000 !important;
    }

    /* Style for st.text_area label (Paste SMILES string(s):) */
    div[data-testid="stTextArea"] label p {
        font-size: 16px !important;
        font-weight: normal !important;
        font-family: zumila;
        color: #000000 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Choose model with radio button
    model_choice = st.radio(
        "Select Model:",
        ["LightGBM", "Random Forest"],
        horizontal=True
    )
    st.write("You selected:", model_choice)

    # Custom CSS for text area
    st.markdown("""
    <style>
    /* Style text area */
    [data-testid="stTextArea"] textarea {
        background-color: #f0ffff !important;  /* light cyan */
        color: #003333 !important;             /* dark teal text */
        font-size: 20px !important;
        font-family: "Courier New", monospace !important;
        border: 2px solid #006666 !important;
        border-radius: 0px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # SMILES input
    smiles_input = st.text_area("Paste SMILES string(s):", key="smiles_input")

    # Example text changes depending on model
    if "LightGBM" in model_choice:
        st.markdown(
            '<p class="example-text">Example: CC1=CC=C(C=C1)N2C(=O)N=C(S2)NC3=CC=CC=C3</p>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<p class="example-text">Example: COC1=CC=CC=C1O</p>',
            unsafe_allow_html=True
        )

    # Custom CSS for file uploader
    st.markdown("""
    <style>
    /* Style file uploader box */
    [data-testid="stFileUploader"] section {
        background-color: #e6ffff !important;  /* light cyan background */
        border: 2px dashed #006666 !important; /* dashed teal border */
        border-radius: 10px !important;
        padding: 10px !important;
    }

    /* Style text inside uploader */
    [data-testid="stFileUploader"] label {
        color: #003333 !important;
        font-weight: 600 !important;
        font-size: 16px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # File uploader
    uploaded = st.file_uploader("Or upload a .smi file", type=["smi"], key="file_upload")

    # Prediction button
    if st.button("Run Prediction"):
        if "LightGBM" in model_choice:
            run_prediction("Model 1", smiles_input, uploaded)
        else:
            run_prediction("Model 2", smiles_input, uploaded)

# ------------------------------
# CONVERTER
# ------------------------------
with tab_con:
    st.markdown("""
    <style>
    .converter-box {
        background: #ffffff;
        
        width: 100%;
    }
    .converter-text {
        font-size: 16px;
        line-height: 1.6;
        text-align: left;
    }
    .converter-title {
        font-size: 30px;
        text-align: center;
        color: #006666;
        font-weight: bold;
        margin-bottom: 12px;
    }
    /* spacing fix */
    div[data-testid="stRadio"] {
        margin-top: 5px;
    }
    /* Center radio & inputs */
    .converter-container {
        max-width: 600px;
        margin: auto;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="converter-box">
            <div class="converter-title">Converter Module</div>
            <div class="converter-text">
                This module was developed for bidirectional conversion between IC50 (in M) and pIC50.
                Users can select the conversion type and obtain the result. For more information,
                please refer to the Help page.
            </div>
        </div>
        """, unsafe_allow_html=True)

    conversion_type = st.radio("Select conversion type:", ["pIC₅₀ ➝ IC₅₀", "IC₅₀ ➝ pIC₅₀"])

    if conversion_type == "pIC₅₀ ➝ IC₅₀":
        pic50_value = st.text_input("Enter pIC₅₀ value:")
        if st.button("Convert to IC₅₀"):
            try:
                val = float(pic50_value)
                ic50_value = pIC50_to_IC50(val)
                st.success(f"IC₅₀ value: {ic50_value:.6e} M")
            except:
                st.error("Invalid input.")
    else:
        ic50_value = st.text_input("Enter IC₅₀ value (in M):")
        if st.button("Convert to pIC₅₀"):
            try:
                val = float(ic50_value)
                if val > 0:
                    pic50_value = IC50_to_pIC50(val)
                    st.success(f"pIC₅₀ value: {pic50_value:.4f}")
                else:
                    st.error("IC₅₀ must be > 0")
            except:
                st.error("Invalid input.")

# ------------------------------
# DATASET
# ------------------------------
with tab_data:
    st.markdown("""
    <style>
    /* Base button style */
    div.stButton > button {
        background-color: #e0ffff !important;   /* light cyan */
        color: #004d4d !important;              /* dark text */
        border: 2px solid #004d4d !important;
        border-radius: 8px;
        font-weight: bold;
        padding: 8px 20px;
        transition: all 0.2s ease-in-out;
    }
    /* Hover effect */
    div.stButton > button:hover {
        background-color: #66b2b2 !important;   /* medium teal */
        color: white !important;
    }
    /* Active/Clicked effect */
    div.stButton > button:active {
        background-color: #004d4d !important;   /* dark teal */
        color: #ffffff !important;
        border: 2px solid #002626 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""           
        <div class="prediction-title">Dataset of MDM2InPred</div>
        """, unsafe_allow_html=True)

    models = ["LightGBM", "Random Forest"]
    files = {
        "LightGBM": {
            "Training Set": "static/lightgbm_train_set.csv",
            "Test Set": "static/lightgbm_test_set.csv",
            "External Validation Set": "static/predicted_activity_external_lightgbm.csv"
        },
        "Random Forest": {
            "Training Set": "static/train_set.csv",
            "Test Set": "static/test_set.csv",
            "External Validation Set": "static/validation_set.csv"
        }
    }

    # Table styling
    st.markdown(
        """
        <style>
        .dataset-table th {
            background-color: #006666;
            color: white;
            padding: 10px;
            text-align: center;
        }
        .dataset-table td {
            text-align: center;
            padding: 8px;
            border: 1px solid #ddd;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Make table layout with buttons
    for model in models:
        st.write(f"**{model}**")
        cols = st.columns(3)
        for i, dataset in enumerate(["Training Set", "Test Set", "External Validation Set"]):
            file_path = files[model][dataset]
            if cols[i].button(f"📂 {dataset}", key=f"{model}-{dataset}"):
                if os.path.exists(file_path):
                    st.subheader(f"📂 {model} - {dataset}")
                    df = pd.read_csv(file_path)
                    st.dataframe(df, use_container_width=True)
                    st.markdown(
                        """
                        <style>
                            [data-testid="stElementToolbar"] {
                                display: none;
                            }
                        </style>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.error(f"❌ File not found: {file_path}")

# ------------------------------
# HELP
# ------------------------------
with tab_help:
    st.markdown("""
    <style>
    .prediction-box {
        background: #ffffff;
        height: 50px;
        width: 1270px;
        padding: 0px;
        margin-bottom: 15px;
        margin-top: 0px;
    }
    .help-title {
        font-family: book antiqua;
        font-size: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="prediction-box">
            <div class="prediction-text">
                This dashboard is designed to provide an interactive interface for researchers to screen chemical 
                compounds as potential MDM2 inhibitors and non-inhibitors virtually. It includes a prediction module 
                for the prediction of MDM2 inhibitors and non-inhibitors, and an additional module for bidirectional 
                conversion between IC50 and pIC50. The following video tutorial demonstrates how to navigate the 
                dashboard and access its features.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown("")
    # ---- Add video ----
    st.video("video.mp4")

# ------------------------------
# CONTACT
# ------------------------------
with tab_contact:

    # ---------- Helper: Convert Image to Base64 ----------
    def get_base64_image(image_path):
        with open(image_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()

    # ---------- Styles ----------
    st.markdown("""
    <style>
    .section-title {
        font-size: 30px;
        text-align: center;
        color: #006666;
        font-weight: 700;
        margin: 20px 0 30px 0;
        font-family: 'Segoe UI', sans-serif;
    }
    .profile-card {
        background: #f9f9f9;
        border-radius: 15px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0,0,0,0.08);
        margin: 10px;
    }
    .profile-card img {
        width: 120px;
        height: 120px;
        border-radius: 50%;
        object-fit: cover;
        margin-bottom: 10px;
        border: 3px solid #006666;
    }
    .profile-name {
        font-size: 16px;
        font-weight: 600;
        color: #006666;
        margin-bottom: 3px;
    }
    .profile-role {
        font-size: 14px;
        color: #333;
        margin-bottom: 3px;
    }
    .profile-email {
        font-size: 13px;
        color: #555;
    }
    </style>
    """, unsafe_allow_html=True)

    # ---------- Contact Us Title ----------
    st.markdown("<div class='section-title'>Contact Us</div>", unsafe_allow_html=True)

    # ---------- Head Profile ----------
    #head_img = get_base64_image("images/head.jpg")

    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown(f"""
        <div class="profile-card">
            <div class="profile-name">Dr. Sarfaraz Alam</div>
            <div class="profile-role">Associate Professor</div>
            <div class="profile-role">CADDynOmics Lab</div>
            <div class="profile-role">Institute of Advanced Research, The University for Innovation, Gandhinagar</div>
            <div class="profile-email">sarfaraz.alam@iar.ac.in, </div>
        </div>
        """, unsafe_allow_html=True)

    # ---------- Our Team Title ----------
    st.markdown("<div class='section-title'>Our Team</div>", unsafe_allow_html=True)

    # ---------- Team Members ----------
    team = [
        ("images/1.jpg", "Zarnalipi soren", "P.HD ", "riya@example.com"),
        ("images/riyaa.jpg", "Riya Patel", "M.Sc data science", "riya20.surat@gmail.com"),
        ("images/pranjal.jpeg", "Pranjal Oza", "M.Sc data science", "pranjaloza7@gmail.com"),
        ("images/meet.jpeg", "Meet Bhayani", "M.Sc data science", "meetmbhayani@gmail.com"),
        ("images/raish.jpeg", "Raishbhai Mansuri", "M.Sc data science", "raishmansuri2003@gmail.com"),
    ]

    for i in range(0, len(team), 3):   # 3 per row
        row = team[i:i+3]
        cols = st.columns(len(row))
        for col, (img, name, role, email) in zip(cols, row):
            with col:
                img_base64 = get_base64_image(img)
                st.markdown(f"""
                <div class="profile-card">
                    <img src="data:image/png;base64,{img_base64}">
                    <div class="profile-name">{name}</div>
                    <div class="profile-role">{role}</div>
                    <div class="profile-email">{email}</div>
                </div>
                """, unsafe_allow_html=True)
# ------------------------------
# CHATBOT
# ------------------------------
with tab_chat:

    st.markdown("""
    <style>
    .chat-title {
        font-size: 26px;
        text-align: center;
        color: #006666;
        font-weight: bold;
        margin-top: 10px;
        margin-bottom: 5px;
        font-family: 'Segoe UI', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='chat-title'>MDM2InPred Assistant</div>", unsafe_allow_html=True)

    # 🔹 Short description instead of big box
    st.markdown(
        """
        <p style="font-size:16px; line-height:1.6; text-align:justify;">
        This assistant is designed to help you understand and use the MDM2InPred dashboard.
        You can ask questions about:
        </p>
        <ul style="font-size:16px; line-height:1.6;">
            <li>How to use the <b>Prediction</b> module (SMILES input, model selection, CSV output).</li>
            <li>How the <b>Converter</b> works for IC₅₀ and pIC₅₀ values.</li>
            <li>What information is available in the <b>Dataset</b> tab.</li>
            <li>General concepts such as MDM2, p53, inhibitors, IC₅₀, pIC₅₀, LightGBM and Random Forest.</li>
        </ul>
        <p style="font-size:16px; line-height:1.6; text-align:justify;">
        Type your question below and the assistant will respond using the information and context
        of this dashboard.
        </p>
        """,
        unsafe_allow_html=True
    )

    # ---- Show chat history in simple list ----
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("#### Conversation")
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"**You:** {msg['text']}")
            else:
                st.markdown(f"**Assistant:** {msg['text']}")

    st.markdown("---")

    # ---- User input ----
    user_input = st.text_input("Type your question here:", key="chat_input")

    col_send, col_clear = st.columns([1, 1])
    with col_send:
        send_btn = st.button("Send")
    with col_clear:
        clear_btn = st.button("Clear Chat")

    # Clear just resets history – button itself triggers rerun
    if clear_btn:
        st.session_state.chat_history = []

    # On send: call Cohere
    if send_btn and user_input.strip():
        text = user_input.strip()

        # 1) store user message
        st.session_state.chat_history.append({"role": "user", "text": text})

        # 2) get LLM answer
        bot_answer = chatbot_reply(text)

        # 3) store bot message
        st.session_state.chat_history.append({"role": "bot", "text": bot_answer})