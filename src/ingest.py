import fitz  # PyMuPDF
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from typing import List, Dict, Tuple
import logging
from tqdm import tqdm
from utils import chunk_text

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DATA_DIR = "data/"
STORAGE_DIR = "storage/"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF file with error handling."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num, page in enumerate(doc):
            try:
                page_text = page.get_text("text")
                if page_text.strip():  # Only add non-empty pages
                    text += f"\n--- Page {page_num + 1} ---\n"
                    text += page_text + "\n"
            except Exception as e:
                logger.warning(f"Error extracting text from page {page_num + 1} of {pdf_path}: {e}")
        doc.close()
        return text
    except Exception as e:
        logger.error(f"Error opening PDF {pdf_path}: {e}")
        return ""

def process_pdfs() -> Tuple[List[str], List[Dict]]:
    """Process all PDFs in the data directory and return texts and metadata."""
    texts, metadata = [], []
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {DATA_DIR}")
        return texts, metadata
    
    logger.info(f"Processing {len(pdf_files)} PDF files...")
    
    for fname in tqdm(pdf_files, desc="Processing PDFs"):
        path = os.path.join(DATA_DIR, fname)
        raw_text = extract_text_from_pdf(path)
        
        if not raw_text.strip():
            logger.warning(f"No text extracted from {fname}")
            continue
            
        try:
            chunks = chunk_text(raw_text)
            if chunks:
                texts.extend(chunks)
                # Enhanced metadata with chunk information
                for i, chunk in enumerate(chunks):
                    metadata.append({
                        "source": fname,
                        "chunk_id": i,
                        "chunk_length": len(chunk),
                        "file_path": path
                    })
                logger.info(f"Processed {fname}: {len(chunks)} chunks")
            else:
                logger.warning(f"No chunks created from {fname}")
        except Exception as e:
            logger.error(f"Error chunking text from {fname}: {e}")
    
    return texts, metadata

def create_embeddings(texts: List[str], model_name: str = EMBEDDING_MODEL) -> np.ndarray:
    """Create embeddings for the given texts."""
    if not texts:
        raise ValueError("No texts provided for embedding creation")
    
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    
    logger.info(f"Creating embeddings for {len(texts)} text chunks...")
    embeddings = model.encode(
        texts, 
        convert_to_numpy=True, 
        show_progress_bar=True,
        batch_size=32  # Adjust based on memory constraints
    )
    
    logger.info(f"Created embeddings with shape: {embeddings.shape}")
    return embeddings

def build_faiss_index() -> None:
    """Build and save FAISS index with embeddings and metadata."""
    try:
        # Process PDFs
        texts, metadata = process_pdfs()
        
        if not texts:
            logger.error("No texts found to index. Please check your PDF files and data directory.")
            return
        
        # Create embeddings
        embeddings = create_embeddings(texts)
        
        # Build FAISS index
        logger.info("Building FAISS index...")
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings.astype(np.float32))
        
        # Save index and metadata
        os.makedirs(STORAGE_DIR, exist_ok=True)
        
        index_path = os.path.join(STORAGE_DIR, "faiss.index")
        metadata_path = os.path.join(STORAGE_DIR, "metadata.pkl")
        
        faiss.write_index(index, index_path)
        logger.info(f"FAISS index saved to: {index_path}")
        
        with open(metadata_path, "wb") as f:
            pickle.dump({
                "texts": texts, 
                "metadata": metadata,
                "embedding_model": EMBEDDING_MODEL,
                "index_info": {
                    "total_chunks": len(texts),
                    "embedding_dim": dim,
                    "pdf_count": len(set(m["source"] for m in metadata))
                }
            }, f)
        logger.info(f"Metadata saved to: {metadata_path}")
        
        # Print summary
        unique_sources = set(m["source"] for m in metadata)
        logger.info("✅ FAISS index built successfully!")
        logger.info(f"   📄 PDFs processed: {len(unique_sources)}")
        logger.info(f"   📝 Text chunks: {len(texts)}")
        logger.info(f"   🔢 Embedding dimensions: {dim}")
        logger.info(f"   💾 Storage location: {STORAGE_DIR}")
        
    except Exception as e:
        logger.error(f"Error building FAISS index: {e}")
        raise

def verify_index() -> None:
    """Verify that the built index can be loaded correctly."""
    try:
        index_path = os.path.join(STORAGE_DIR, "faiss.index")
        metadata_path = os.path.join(STORAGE_DIR, "metadata.pkl")
        
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            logger.error("Index files not found. Please build the index first.")
            return
        
        # Load and verify
        index = faiss.read_index(index_path)
        with open(metadata_path, "rb") as f:
            data = pickle.load(f)
        
        logger.info("✅ Index verification successful!")
        logger.info(f"   Index size: {index.ntotal} vectors")
        logger.info(f"   Metadata entries: {len(data['texts'])}")
        
        if 'index_info' in data:
            info = data['index_info']
            logger.info(f"   PDFs indexed: {info['pdf_count']}")
            logger.info(f"   Embedding model: {data.get('embedding_model', 'Unknown')}")
        
    except Exception as e:
        logger.error(f"Error verifying index: {e}")

if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(STORAGE_DIR, exist_ok=True)
    
    # Build the index
    build_faiss_index()
    
    # Verify the built index
    verify_index()