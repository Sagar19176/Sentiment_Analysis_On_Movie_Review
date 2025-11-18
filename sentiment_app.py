import streamlit as st
import pandas as pd
import numpy as np
import re
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Sentiment Analysis System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM STYLING ---
st.markdown("""
    <style>
    .main {
        background-color: #f4f6f9; /* Slightly darker off-white for contrast */
    }
    .stButton>button {
        width: 100%;
        background-color: #007bff;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        color: white;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    
    /* METRIC CARD STYLING - FIXED */
    .metric-card {
        background-color: #ffffff !important; /* Force solid white */
        border: 1px solid #e1e4e8; /* Clean subtle border */
        padding: 20px;
        border-radius: 12px; /* Softer corners */
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); /* Clean, soft shadow (not milky) */
        text-align: center;
        margin-bottom: 10px;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
    }
    
    /* Explicit text colors to avoid milky transparency issues */
    .metric-card h3 {
        color: #5f6368 !important; /* Dark grey label */
        font-size: 1rem;
        font-weight: 500;
        margin: 0;
        font-family: 'Segoe UI', sans-serif;
    }
    .metric-card h2 {
        color: #1a73e8 !important; /* Sharp Blue value */
        font-size: 2rem;
        font-weight: 700;
        margin: 5px 0;
        font-family: 'Segoe UI', sans-serif;
    }
    .metric-card p {
        color: #9aa0a6 !important; /* Light grey caption */
        font-size: 0.85rem;
        margin: 0;
    }

    h1, h2, h3 {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 1. DATA COLLECTION MODULE ---
@st.cache_data
def load_data():
    """
    Loads the IMDB dataset. 
    If the CSV file exists, it uses that. Otherwise, it uses synthetic data.
    """
    try:
        # READ THE CSV FILE
        # We use only the first 5,000 rows to keep training fast for the presentation
        # If you want all 50k, remove 'nrows=5000' (but it will take ~30s to train)
        df = pd.read_csv('IMDB_Dataset.csv').head(5000)
        
        # Standardize column names just in case
        df.columns = [c.lower() for c in df.columns] 
        
        # Ensure we have the right columns (rename if necessary)
        if 'review' not in df.columns or 'sentiment' not in df.columns:
             st.error("Error: Dataset must have 'review' and 'sentiment' columns.")
             return pd.DataFrame()
             
        return df

    except FileNotFoundError:
        st.warning("⚠️ 'IMDB_Dataset.csv' not found. Using synthetic demo data instead.")
        
        # Fallback Synthetic Data
        data = {
            'review': [
                "I absolutely loved this movie! The acting was fantastic.",
                "What a waste of time. The plot was terrible.",
                "A masterpiece. Beautifully filmed and directed.",
                "I fell asleep halfway through. Boring.",
                "Great visual effects but the story was lacking.",
                "Worst movie I have ever seen. Do not watch.",
                "An emotional rollercoaster. Highly recommended.",
                "The characters were shallow and unlikable.",
                "Brilliant performance by the lead actor.",
                "Script was weak and predictable.",
                "I enjoyed every minute of it. Pure entertainment.",
                "Disappointing conclusion to the trilogy.",
                "Funny, witty, and charming.",
                "Awful directing and poor editing.",
                "A classic that will be remembered for years.",
                "Horrible pacing, it felt like it went on forever.",
                "Truly inspiring and heartwarming.",
                "I regret paying money to see this.",
                "Cinematography was stunning, 10/10.",
                "Garbage. Absolute garbage."
            ] * 5, 
            'sentiment': [
                "positive", "negative", "positive", "negative", "negative", "negative",
                "positive", "negative", "positive", "negative", "positive", "negative",
                "positive", "negative", "positive", "negative", "positive", "negative",
                "positive", "negative"
            ] * 5
        }
        return pd.DataFrame(data)

# --- 2. PREPROCESSING MODULE ---
def clean_text(text):
    """
    Text Cleaning Pipeline:
    1. Lowercase
    2. Remove HTML
    3. Remove special characters
    """
    text = text.lower()
    text = re.sub('<.*?>', '', text) # Remove HTML tags
    text = re.sub('[^a-zA-Z\s]', '', text) # Remove special characters/numbers
    return text

# --- 3. TRAINING MODULE ---
def train_model(df):
    # Preprocessing
    df['clean_review'] = df['review'].apply(clean_text)
    
    # Feature Extraction (TF-IDF)
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    X = tfidf.fit_transform(df['clean_review']).toarray()
    y = df['sentiment']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model Training (Logistic Regression)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    return model, tfidf, acc, y_test, y_pred

# --- APP LOGIC ---

# Load Data
df = load_data()

# Train Model (Cached so it doesn't retrain on every click)
if 'model' not in st.session_state:
    with st.spinner('Initializing System... Cleaning Data... Training Model...'):
        model, tfidf, accuracy, y_test, y_pred = train_model(df)
        st.session_state['model'] = model
        st.session_state['tfidf'] = tfidf
        st.session_state['accuracy'] = accuracy
        st.session_state['y_test'] = y_test
        st.session_state['y_pred'] = y_pred
        time.sleep(0.5) 

# Retrieve from session
model = st.session_state['model']
tfidf = st.session_state['tfidf']
acc = st.session_state['accuracy']

# --- UI LAYOUT ---

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1200px-Python-logo-notext.svg.png", width=50)
    st.title("Project Modules")
    st.info("Sentiment Analysis System v1.3")
    
    menu = st.radio(
        "Navigation", 
        ["Dashboard", "Model Prediction", "Visualizations", "About"]
    )
    
    st.markdown("---")
    st.markdown("### System Status")
    st.success(f"Model Trained Successfully")
    st.write(f"**Algorithm:** Logistic Regression")
    st.write(f"**Accuracy:** {acc*100:.2f}%")

# Main Content
if menu == "Dashboard":
    st.title("🎬Movie Review Sentiment Analysis Dashboard")
    st.markdown("#### Welcome, Sagar Kumar (231950036)")
    st.write("This centralized dashboard allows you to analyze movie reviews using Machine Learning.")
    
    # UPDATED METRIC CARDS SECTION
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <h3>Total Reviews</h3>
            <h2>50k+</h2>
            <p>IMDB Dataset Size</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <h3>Model Accuracy</h3>
            <h2>{acc*100:.1f}%</h2>
            <p>Test Set Performance</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='metric-card'>
            <h3>Classification</h3>
            <h2>Binary</h2>
            <p>Positive / Negative</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    st.subheader("Recent Dataset Entries")
    st.dataframe(df.head(5), use_container_width=True)

elif menu == "Model Prediction":
    st.title("🤖 Real-Time Prediction")
    st.write("Enter a movie review below to test the model instantly.")
    
    user_input = st.text_area("Review Text", height=150, placeholder="Type something like: 'The movie was absolutely fantastic!'")
    
    col1, col2 = st.columns([1, 2])
    
    if col1.button("Analyze Sentiment"):
        if user_input:
            # Preprocess
            cleaned = clean_text(user_input)
            # Vectorize
            vec = tfidf.transform([cleaned])
            # Predict
            prediction = model.predict(vec)[0]
            proba = model.predict_proba(vec)[0]
            
            st.markdown("---")
            if prediction == "positive":
                st.success(f"## Result: POSITIVE 😊")
                st.progress(proba[1])
                st.write(f"Confidence Score: **{proba[1]*100:.2f}%**")
            else:
                st.error(f"## Result: NEGATIVE 😠")
                st.progress(proba[0])
                st.write(f"Confidence Score: **{proba[0]*100:.2f}%**")
        else:
            st.warning("Please enter some text first.")

elif menu == "Visualizations":
    st.title("📊 Performance Analytics")
    st.write("Visual representation of the model's performance and data distribution.")
    
    tab1, tab2 = st.tabs(["Confusion Matrix", "Class Distribution"])
    
    with tab1:
        st.subheader("Confusion Matrix Heatmap")
        st.write("This chart shows how many reviews were correctly vs incorrectly classified.")
        fig_cm, ax = plt.subplots(figsize=(6, 4))
        cm = confusion_matrix(st.session_state['y_test'], st.session_state['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                    xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        st.pyplot(fig_cm)
        
    with tab2:
        st.subheader("Data Balance Check")
        st.write("Ensuring the dataset has an equal number of positive and negative reviews.")
        fig_dist, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x='sentiment', data=df, palette='viridis', ax=ax)
        st.pyplot(fig_dist)

elif menu == "About":
    st.title("ℹ️ About the Project")
    st.markdown("""
    **Project Title:** Sentiment Analysis on Movie Reviews  
    **Submitted By:** Sagar Kumar (231950036)  
    **Submitted To:** Dr. Swati Bansal  
    **Department:** CSE (AI & ML)
    
    ---
    **Technical Details:**
    * **Language:** Python 3.9+
    * **Libraries:** Scikit-Learn, Pandas, NLTK, Streamlit
    * **Algorithm:** Logistic Regression
    * **Feature Extraction:** TF-IDF (Term Frequency - Inverse Document Frequency)
    """)

# Footer
st.markdown("---")

st.markdown(f"<center>Developed by Sagar Kumar | © 2025 MRSPTU</center>", unsafe_allow_html=True)
