import streamlit as st
from transformers import pipeline
from newspaper import Article
import time
import nltk

# 1. Essential Setup: Downloads required logic for reading articles
@st.cache_resource
def setup_tools():
    try:
        nltk.download('punkt')
        nltk.download('punkt_tab')
    except:
        pass

setup_tools()

# 2. Page Config
st.set_page_config(page_title="AI Article Summarizer", page_icon="📝")

# 3. Stable Model Loading
# We use 'sshleifer/distilbart-cnn-12-6' because it fits in Streamlit's RAM
@st.cache_resource
def load_model():
    # Explicitly naming the task and model prevents 'KeyError'
    return pipeline(task="summarization", model="sshleifer/distilbart-cnn-12-6")

summarizer = load_model()

st.title("🤖 AI Article Summarizer")
st.markdown("Instantly condense news articles into key points.")

# 4. Input Area
url = st.text_input("🔗 Paste News Article URL here:", placeholder="https://www.example.com/news-story")

if st.button("Summarize Now"):
    if url:
        try:
            with st.spinner('AI is reading the article...'):
                # Extracting Text
                article = Article(url)
                article.download()
                article.parse()
                
                if not article.text:
                    st.error("❌ Could not extract text. Try a different news URL.")
                else:
                    # AI Processing
                    # We limit input to 3000 chars to prevent memory crashes
                    start_time = time.time()
                    summary_output = summarizer(article.text[:3000], max_length=130, min_length=30, do_sample=False)
                    summary_text = summary_output[0]['summary_text']
                    processing_time = round(time.time() - start_time, 2)
                    
                    # Display Results
                    st.subheader(f"📄 {article.title}")
                    st.success(summary_text)
                    
                    # Performance Metrics
                    st.write("---")
                    st.info(f"**Stats:** {len(article.text.split())} words distilled to {len(summary_text.split())} words in {processing_time}s")
                    
        except Exception as e:
            st.error(f"⚠️ System Error: {str(e)}")
    else:
        st.warning("⚠️ Please provide a URL first.")

st.sidebar.markdown("### NLP Project Details")
st.sidebar.write("Using DistilBART Transformer")
