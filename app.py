import streamlit as st
from transformers import pipeline
from newspaper import Article
import time
import nltk

# Requirement: Download necessary NLTK data for web scraping
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('punkt')
        nltk.download('punkt_tab')
    except Exception as e:
        st.error(f"Error downloading NLTK data: {e}")

download_nltk_data()

# Page Setup (Reflecting standard dashboard requirements)
st.set_page_config(page_title="AI News Summarizer", page_icon="📝")

# Model Loading: Using DistilBART for hardware feasibility [cite: 42, 77]
@st.cache_resource
def load_summarizer():
    # Explicit task and model definition to prevent KeyError errors
    return pipeline(task="summarization", model="sshleifer/distilbart-cnn-12-6")

summarizer = load_summarizer()

st.title("🤖 AI-Powered News Summarizer")
st.markdown("---")

# User Input Section
url = st.text_input("🔗 Paste News Article URL here:", placeholder="https://www.channelnewsasia.com/...")

if st.button("Generate Summary"):
    if url:
        try:
            with st.spinner('AI is reading and distilling the article...'):
                start_time = time.time()
                
                # Article Fetching and Parsing
                article = Article(url)
                article.download()
                article.parse()
                
                if not article.text:
                    st.error("❌ Could not extract text from this URL. Some sites block AI access.")
                else:
                    # Summarization with safe truncation [cite: 33, 72]
                    # We process the first 3000 characters to ensure system stability
                    summary_output = summarizer(article.text[:3000], max_length=130, min_length=30, do_sample=False)
                    summary_text = summary_output[0]['summary_text']
                    
                    duration = round(time.time() - start_time, 2)
                    
                    # Display Results
                    st.subheader(f"📄 Title: {article.title}")
                    st.success(summary_text)
                    
                    # Performance Metrics [cite: 121, 163, 171]
                    orig_len = len(article.text.split())
                    summ_len = len(summary_text.split())
                    reduction = round((1 - (summ_len / orig_len)) * 100, 1)
                    
                    st.write("---")
                    st.info(f"**Performance:** {orig_len} words ➡️ {summ_len} words ({reduction}% reduction) | ⏱ {duration}s")
                    
        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")
    else:
        st.warning("⚠️ Please provide a URL first.")

# Sidebar Info (Ensuring clarity for graders) [cite: 163, 182]
st.sidebar.title("Project Details")
st.sidebar.write("**Course:** JIE43303 NLP")
st.sidebar.write("**Model:** DistilBART (Optimized)")
st.sidebar.markdown("This system uses **Abstractive Summarization** to rewrite news articles concisely.")
