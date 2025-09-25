import streamlit as st
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from datetime import datetime
import time

# ----------- Settings -----------
MAX_WORDS = 1000
MAX_LEN = 100
TOKENIZER_PATH = 'tokenizer.pkl'
MODEL_PATH = 'sentiment_model.h5'

# ----------- Helper Functions -----------
def text_cleaning(text):
    text = text.lower()
    text = re.sub(r"[^A-Za-z]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

@st.cache_resource
def load_tokenizer(path):
    try:
        with open(path, 'rb') as handle:
            tokenizer = pickle.load(handle)
        return tokenizer
    except FileNotFoundError:
        st.error(f"Tokenizer file not found at {path}. Please ensure the file exists.")
        return None

@st.cache_resource
def load_model(path):
    try:
        model = tf.keras.models.load_model(path)
        return model
    except:
        st.error(f"Model file not found at {path}. Please ensure the file exists.")
        return None

def predict_sentiment(review, model, tokenizer):
    if not model or not tokenizer:
        return None, None
    
    clean_review = text_cleaning(review)
    seq = tokenizer.texts_to_sequences([clean_review])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    pred = model.predict(padded, verbose=0)
    sentiment = "Positive" if pred >= 0.5 else "Negative"
    confidence = float(pred[0][0])
    return sentiment, confidence

# ----------- Sample Reviews -----------
SAMPLE_REVIEWS = {
    "Positive Example": "This product is absolutely amazing! The quality is outstanding and it arrived quickly. I'm very satisfied with my purchase and would definitely recommend it to others.",
    "Negative Example": "Terrible product. It broke after just one day of use. Poor quality materials and awful customer service. Complete waste of money."
}

# ----------- Streamlit UI -----------
st.set_page_config(
    page_title="Amazon Review Sentiment Analyzer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 3rem;
    }
    .result-positive {
        background: linear-gradient(90deg, #4ecdc4, #44a08d);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .result-negative {
        background: linear-gradient(90deg, #ff6b6b, #ee5a52);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .stats-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
        color: #333333;
    }
    /* Fix only analysis stats text visibility */
    div[data-testid="column"] .metric-container label {
        color: #333333 !important;
    }
    div[data-testid="column"] .metric-container [data-testid="metric-container"] {
        color: #333333 !important;
    }
    div[data-testid="column"] h3 {
        color: #333333 !important;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🛍️ Amazon Review Sentiment Analyzer</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Analyze the sentiment of Amazon product reviews using AI</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📊 About This Tool")
    st.write("""
    This AI-powered tool analyzes Amazon product reviews to determine if they express positive or negative sentiment.
    
    **Features:**
    - Real-time sentiment analysis
    - Confidence scoring
    - Interactive visualizations
    - Sample reviews to try
    """)
    
    st.header("🎯 How to Use")
    st.write("""
    1. Enter or paste a review in the text area
    2. Click 'Analyze Sentiment'
    3. View the results with confidence scores
    4. Try the sample reviews below!
    """)
    
    st.header("📝 Try Sample Reviews")
    for name, review in SAMPLE_REVIEWS.items():
        if st.button(f"Load {name}", key=f"btn_{name}"):
            st.session_state.review_input = review  # directly update text area



# ----------- Main content
col1, col2 = st.columns([2, 1])

# Ensure session state key exists
if "review_input" not in st.session_state:
    st.session_state.review_input = ""

with col1:
    st.header("📝 Enter Review Text")
    def clear_text():
        st.session_state.review_input = ""

    # Text area is now bound directly to session_state
    review_text = st.text_area(
        "Paste your Amazon review here:",
        key="review_input",
        height=150,
        placeholder="Enter the review text you want to analyze...",
        help="Paste any Amazon product review to analyze its sentiment"
    )
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
    
    with col_btn1:
        analyze_btn = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)
    
    with col_btn2:
        st.button("🗑️ Clear Text", use_container_width=True, on_click=clear_text)


with col2:
    st.header("📈 Analysis Stats")
    
    # Initialize session state for stats
    if 'total_analyses' not in st.session_state:
        st.session_state.total_analyses = 0
    if 'positive_count' not in st.session_state:
        st.session_state.positive_count = 0
    if 'negative_count' not in st.session_state:
        st.session_state.negative_count = 0
    
    col_stat1, col_stat2 = st.columns(2)
    
    with col_stat1:
        st.metric(
            label="Total Analyses", 
            value=st.session_state.total_analyses,
            delta=None
        )
        st.metric(
            label="Positive Reviews", 
            value=st.session_state.positive_count,
            delta=None
        )
    
    with col_stat2:
        st.metric(
            label="Negative Reviews", 
            value=st.session_state.negative_count,
            delta=None
        )
        
        if st.session_state.total_analyses > 0:
            accuracy = max(st.session_state.positive_count, st.session_state.negative_count) / st.session_state.total_analyses * 100
            st.metric(
                label="Dominant Sentiment", 
                value=f"{accuracy:.1f}%",
                delta=None
            )

# Analysis section
if analyze_btn and review_text.strip():
    with st.spinner("🤖 Analyzing sentiment..."):
        # Simulate processing time for better UX
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        
        # Load model and tokenizer
        tokenizer = load_tokenizer(TOKENIZER_PATH)
        model = load_model(MODEL_PATH)
        
        if tokenizer and model:
            sentiment, confidence = predict_sentiment(review_text, model, tokenizer)
            
            if sentiment and confidence is not None:
                # Update stats
                st.session_state.total_analyses += 1
                if sentiment == "Positive":
                    st.session_state.positive_count += 1
                else:
                    st.session_state.negative_count += 1
                
                # Display results
                st.header("🎯 Analysis Results")
                
                # Result banner
                if sentiment == "Positive":
                    st.markdown(f"""
                    <div class="result-positive">
                        😊 POSITIVE SENTIMENT<br>
                        Confidence: {confidence*100:.1f}%
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-negative">
                        😞 NEGATIVE SENTIMENT<br>
                        Confidence: {(1-confidence)*100:.1f}%
                    </div>
                    """, unsafe_allow_html=True)
                
                # Additional insights
                st.subheader("💡 Insights")
                
                confidence_level = "High" if (confidence > 0.8 or confidence < 0.2) else "Medium" if (confidence > 0.6 or confidence < 0.4) else "Low"
                
                col_insight1, col_insight2, col_insight3 = st.columns(3)
                
                with col_insight1:
                    st.metric("Sentiment", sentiment)
                
                with col_insight2:
                    st.metric("Confidence Level", confidence_level)
                
                with col_insight3:
                    st.metric("Word Count", len(review_text.split()))
                
                # Cleaned text preview
                with st.expander("🔍 View Processed Text"):
                    cleaned = text_cleaning(review_text)
                    st.code(cleaned, language="text")
        
        progress_bar.empty()

elif analyze_btn and not review_text.strip():
    st.warning("⚠️ Please enter some review text to analyze!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>🤖 Powered by TensorFlow & Streamlit | Built with ❤️ for Amazon Review Analysis</p>
    <p><small>This tool uses machine learning to analyze sentiment. Results may vary based on text complexity and context.</small></p>
</div>
""", unsafe_allow_html=True)
