import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import logging
from typing import List

from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load GROQ API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Streamlit page config
st.set_page_config(
    page_title="Multi-PDF AI Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📄 Multi-PDF AI Assistant")
st.markdown("Upload multiple PDF files and ask questions about their content using AI.")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    chunk_size = st.slider("Chunk Size", 500, 2000, 1000, 100)
    chunk_overlap = st.slider("Chunk Overlap", 50, 300, 100, 25)
    
    st.header("ℹ️ Instructions")
    st.markdown("""
    1. Upload one or more PDF files
    2. Wait for processing
    3. Ask questions about their content
    """)

def validate_api_key() -> bool:
    if not GROQ_API_KEY:
        st.error("⚠️ GROQ_API_KEY not found. Set it in environment variables.")
        return False
    return True

def extract_text_from_pdfs(files: List) -> List[str]:
    """Extract text from PDFs and split into chunks"""
    all_chunks = []
    with st.spinner("📖 Extracting text from PDFs..."):
        progress_bar = st.progress(0)
        for i, file in enumerate(files):
            try:
                file.seek(0)
                pdf = PdfReader(file)
                file_text = ""
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        file_text += f"\n--- {file.name} - Page {page_num+1} ---\n"
                        file_text += page_text + "\n"
                if file_text.strip():
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                    chunks = text_splitter.split_text(file_text)
                    all_chunks.extend(chunks)
                    st.success(f"✅ Extracted {len(chunks)} chunks from {file.name}")
                else:
                    st.warning(f"⚠️ No text found in {file.name}")
            except Exception as e:
                logger.error(f"Error processing {file.name}: {e}")
                st.error(f"❌ Failed to process {file.name}")
            progress_bar.progress((i+1)/len(files))
    return all_chunks

def answer_query(user_question: str, chunks: List[str], k: int = 5) -> str:
    """Answer question using top-k chunks and Groq"""
    try:
        # For simplicity, pick first k chunks (or you can improve with similarity later)
        context = "\n".join(chunks[:k])
        llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model="llama-3.1-8b-instant",
            temperature=0.1,
            max_tokens=1000
        )
        prompt = f"""Based on the following context, answer the question accurately. If not enough info, say so.

Context:
{context}

Question: {user_question}

Answer:"""
        answer = llm.invoke(prompt)
        return answer.content if hasattr(answer, 'content') else str(answer)
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        st.error("❌ Failed to generate answer")
        return ""

def main():
    if not validate_api_key():
        st.stop()

    st.header("📁 Upload PDF Files")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.success(f"📄 Uploaded {len(uploaded_files)} PDF(s)")
        all_chunks = extract_text_from_pdfs(uploaded_files)
        if all_chunks:
            st.divider()
            st.header("💬 Ask Questions")
            user_question = st.text_input(
                "Ask a question about your PDFs",
                placeholder="e.g., What are the main topics?"
            )
            if user_question:
                answer = answer_query(user_question, all_chunks, k=5)
                if answer:
                    st.divider()
                    st.header("🤖 AI Response")
                    st.markdown(answer)
        else:
            st.error("❌ No text could be extracted.")
    else:
        st.info("👆 Please upload PDF files.")

if __name__ == "__main__":
    main()
