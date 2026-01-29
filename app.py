import streamlit as st
from transformers import pipeline
from newspaper import Article
import time
import nltk

# Force download of required NLTK data for scraping logic
@st.cache_resource
def setup_nltk():
    try:
        nltk.download('punkt')
        nltk.download('punkt_tab')
    except:
        pass

setup_nltk()

# Page Setup
st.set_page_config(page_title="AI Article Summarizer", page_icon="📝")

# Optimized Model Loading (Prevents KeyError and Memory Crashes)
@st.cache_resource
def load_summarizer():
    # We use DistilBART because it is 40% smaller than standard models [cite: 64]
    return pipeline(task="summarization", model="sshleifer/distilbart-cnn-12-6")

summarizer = load_summarizer()

st.title("🤖 AI News Article Summarizer")
st.markdown("Paste any news article URL below to get an instant, AI-generated summary.")

# User Input Section
url = st.text_input("🔗 Paste News Article URL here:", placeholder="https://www.example.com/news...")

if st.button("Generate Summary"):
    if url:
        try:
            with st.spinner('AI is reading and distilling the article...'):
                start_time = time.time()
                
                # Fetch and Parse Article
                article = Article(url)
                article.download()
                article.parse()
                
                if not article.text:
                    st.error("❌ Could not extract text. This site may be blocking automated access.")
                else:
                    # Summarization with safe truncation (first 3000 chars) [cite: 33, 72]
                    # Truncation avoids memory allocation failures (OSError 1455) [cite: 73]
                    summary_output = summarizer(article.text[:3000], max_length=130, min_length=30, do_sample=False)
                    summary_text = summary_output[0]['summary_text']
                    
                    duration = round(time.time() - start_time, 2)
                    
                    # Display Results
                    st.subheader(f"📄 Title: {article.title}")
                    st.success(summary_text)
                    
                    # Performance Metrics (Quantitative measure of efficacy) [cite: 121, 171]
                    orig_len = len(article.text.split())
                    summ_len = len(summary_text.split())
                    reduction = round((1 - (summ_len / orig_len)) * 100, 1)
                    
                    st.write("---")
                    st.info(f"**Performance:** {orig_len} words ➡️ {summ_len} words ({reduction}% reduction) | ⏱ Processing: {duration}s")
                    
        except Exception as e:
            st.error(f"⚠️ System Error: {str(e)}")
    else:
        st.warning("⚠️ Please provide a URL first.")

# Sidebar Info for Clarity
st.sidebar.title("Project Information")
st.sidebar.write("**Task:** JIE43303 NLP")
st.sidebar.write("**Architecture:** Transformer (DistilBART)")
st.sidebar.markdown("This system uses **Abstractive Summarization** to rewrite articles concisely.")
