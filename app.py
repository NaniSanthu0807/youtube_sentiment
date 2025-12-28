import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# ----------------------------------
# UI Styling (YouTube Light Mode)
# ----------------------------------
def apply_light_theme():
    st.markdown("""
    <style>
    /* Main Background - Soft Grey */
    .stApp {
        background-color: #f9f9f9;
        color: #0f0f0f;
    }

    /* Professional White Card */
    div.block-container {
        background-color: #ffffff;
        padding: 2.5rem 3.5rem;
        border-radius: 16px;
        border: 1px solid #e5e5e5;
        margin-top: 40px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
    }

    /* Logo Header */
    .yt-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 25px;
    }

    /* Input Field Styling */
    .stTextArea textarea {
        background-color: #fcfcfc !important;
        color: #0f0f0f !important;
        border: 1px solid #cccccc !important;
        border-radius: 8px !important;
        font-size: 16px;
    }
    .stTextArea textarea:focus {
        border-color: #FF0000 !important;
        box-shadow: 0 0 0 2px rgba(255, 0, 0, 0.1) !important;
    }

    /* Professional YouTube Red Button */
    div.stButton > button {
        background-color: #FF0000;
        color: white;
        border: none;
        padding: 14px 24px;
        border-radius: 30px;
        font-weight: 700;
        transition: all 0.2s ease;
        width: 100%;
        border: 1px solid transparent;
    }
    div.stButton > button:hover {
        background-color: #cc0000;
        box-shadow: 0 4px 10px rgba(204, 0, 0, 0.2);
    }

    /* Analysis Result Card */
    .result-box {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-top: 25px;
        background: #f1f1f1;
        border: 1px solid #e5e5e5;
        animation: slideUp 0.5s ease-out;
    }

    @keyframes slideUp {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Styling for Metric labels */
    .metric-label {
        font-size: 0.85rem;
        color: #606060;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    </style>
    """, unsafe_allow_html=True)

apply_light_theme()

# ----------------------------------
# Logic & NLP Setup
# ----------------------------------
@st.cache_resource
def load_resources():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    with open("log_model.pkl", "rb") as f:
        m = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        v = pickle.load(f)
    return m, v

model, vectorizer = load_resources()
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# ----------------------------------
# UI Content
# ----------------------------------

# Header with Logo
st.markdown("""
    <div class="yt-header">
        <img src="https://upload.wikimedia.org/wikipedia/commons/e/ef/Youtube_logo.png" width="45">
        <h2 style='margin:0; font-family: "YouTube Sans", sans-serif; color: #0f0f0f; font-weight: 800;'>
            Sentiment <span style='color:#FF0000;'>Ai</span>
        </h2>
    </div>
""", unsafe_allow_html=True)

st.write("Scan audience sentiment using high-accuracy NLP classification.")

comment = st.text_area("", placeholder="Enter comment for analysis...", height=130)

if st.button("RUN ANALYSIS"):
    if not comment.strip():
        st.warning("⚠️ Please provide text to analyze.")
    else:
        with st.spinner("Analyzing text patterns..."):
            processed = preprocess_text(comment)
            vectorized = vectorizer.transform([processed])
            
            proba = model.predict_proba(vectorized)[0]
            prediction = model.predict(vectorized)[0]
            confidence = max(proba) * 100

        # Output Card
        color = "#008000" if prediction == 1 else "#CC0000"
        label = "POSITIVE" if prediction == 1 else "NEGATIVE"
        icon = "📈" if prediction == 1 else "📉"

        st.markdown(f"""
            <div class="result-box">
                <span class="metric-label">PREDICTED SENTIMENT</span>
                <h1 style="color: {color}; margin: 5px 0 15px 0; font-weight: 800;">{icon} {label}</h1>
                <hr style="border: 0; border-top: 1px solid #ddd; margin: 15px 0;">
                <span class="metric-label">MODEL CONFIDENCE</span>
                <h2 style="color: #0f0f0f; margin: 5px 0;">{confidence:.2f}%</h2>
            </div>
        """, unsafe_allow_html=True)

        # Confidence Bar (YouTube Red)
        st.markdown("""
            <style>
                .stProgress > div > div > div > div { background-color: #FF0000; }
            </style>
        """, unsafe_allow_html=True)
        st.progress(int(confidence))

st.markdown("""
    <div style="margin-top: 50px; padding-top: 20px; border-top: 1px solid #eee; text-align: center;">
        <p style="margin: 0; color: #b5b5b5; font-size: 12px; letter-spacing: 1px; font-weight: 600;">
            DEVELOPED BY
        </p>
        <p style="margin: 5px 0 0 0; color: #0f0f0f; font-family: 'YouTube Sans', sans-serif; font-size: 16px; font-weight: 800; letter-spacing: 0.5px;">
            BAGADI <span style="color: #FF0000;">SANTHOSH KUMAR</span>
        </p>
        <div style="display: flex; justify-content: center; gap: 15px; margin-top: 10px; opacity: 0.5;">
            <span style="font-size: 10px; color: #606060;">© 2024 ALL RIGHTS RESERVED</span>
        </div>
    </div>
""", unsafe_allow_html=True)