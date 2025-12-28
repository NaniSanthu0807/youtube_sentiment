import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# ----------------------------------
# 1. NLTK Data & Resource Loading
# ----------------------------------
@st.cache_resource
def load_resources():
    # Fix for LookupError: Download data directly on the server
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True) 
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    
    # Load model and vectorizer
    with open("log_model.pkl", "rb") as f:
        m = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        v = pickle.load(f)
    return m, v

model, vectorizer = load_resources()

# ----------------------------------
# 2. UI Styling (YouTube White Mode)
# ----------------------------------
st.set_page_config(page_title="Sentiment AI", page_icon="📺")

st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    div.block-container {
        background-color: #ffffff;
        padding: 3rem;
        border-radius: 20px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        border: 1px solid #edf2f7;
        margin-top: 20px;
    }
    div.stButton > button {
        background-color: #FF0000;
        color: white;
        border-radius: 12px;
        font-weight: 700;
        width: 100%;
        height: 3em;
    }
    .result-card {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        background: #f1f1f1;
        border-top: 5px solid #FF0000;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# ----------------------------------
# 3. Header & Branding
# ----------------------------------
st.markdown("""
    <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 20px;">
        <img src="https://upload.wikimedia.org/wikipedia/commons/e/ef/Youtube_logo.png" width="45">
        <h1 style='margin:0; font-size: 32px;'>Sentiment <span style='color:#FF0000;'>AI</span></h1>
    </div>
    <p style="color: #606060;">High-precision NLP for YouTube Audience Analysis</p>
    <hr>
""", unsafe_allow_html=True)

# ----------------------------------
# 4. Preprocessing Logic
# ----------------------------------
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# ----------------------------------
# 5. Main UI Logic
# ----------------------------------
comment = st.text_area("COMMENT INPUT", placeholder="Paste a YouTube comment here...", height=150)

if st.button("Analyze Sentiment"):
    if not comment.strip():
        st.warning("⚠️ Please enter a comment.")
    else:
        processed = preprocess_text(comment)
        vectorized = vectorizer.transform([processed])
        
        proba = model.predict_proba(vectorized)[0]
        prediction = model.predict(vectorized)[0]
        confidence = max(proba) * 100

        # Display Result
        color = "#10b981" if prediction == 1 else "#ef4444"
        label = "POSITIVE 😊" if prediction == 1 else "NEGATIVE 😠"
        
        st.markdown(f"""
            <div class="result-card" style="border-top-color: {color};">
                <h1 style="color: {color}; margin: 0;">{label}</h1>
                <p style="color: #64748b; font-weight: bold; margin-top: 10px;">Confidence: {confidence:.2f}%</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"<style>.stProgress > div > div > div > div {{ background-color: {color}; }}</style>", unsafe_allow_html=True)
        st.progress(int(confidence))

# ----------------------------------
# 6. Developer Footer
# ----------------------------------
st.markdown(f"""
    <div style="margin-top: 50px; padding-top: 20px; border-top: 1px solid #eee; text-align: center;">
        <p style="margin: 0; color: #b5b5b5; font-size: 11px; letter-spacing: 1px;">DEVELOPED BY</p>
        <p style="margin: 5px 0 0 0; color: #0f0f0f; font-size: 16px; font-weight: 800;">
            BAGADI <span style="color: #FF0000;">SANTHOSH KUMAR</span>
        </p>
    </div>
""", unsafe_allow_html=True)
