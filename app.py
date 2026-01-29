import streamlit as st
from transformers import pipeline
from newspaper import Article
import nltk

# Force download of required NLTK data
try:
    nltk.download('punkt')
    nltk.download('punkt_tab')
except:
    pass

st.set_page_config(page_title="AI News Summarizer", page_icon="📝")

# 1. Load a smaller model to prevent memory crashes (OSError 1455)
@st.cache_resource
def load_summarizer():
    # T5-small is much smaller than BART and very stable for deployment
    return pipeline("summarization", model="t5-small")

summarizer = load_summarizer()

st.title("🤖 AI News Summarizer")
st.write("Summarize any news article URL instantly.")

# 2. Input
url = st.text_input("Paste News URL here:")

if st.button("Summarize"):
    if url:
        try:
            with st.spinner('Reading article...'):
                article = Article(url)
                article.download()
                article.parse()
                
                # Logic to handle short articles
                text = article.text
                if len(text.split()) < 30:
                    st.warning("Article is too short to summarize.")
                else:
                    # AI Processing
                    summary = summarizer(text[:2000], max_length=100, min_length=30, do_sample=False)
                    st.subheader(f"Title: {article.title}")
                    st.success(summary[0]['summary_text'])
                    
                    # Simple Metrics
                    st.info(f"Original: {len(text.split())} words | Summary: {len(summary[0]['summary_text'].split())} words")
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please enter a URL.")
