📄 Multi-PDF AI Assistant

An AI-powered assistant that allows you to upload multiple PDF files and ask questions about their content. It uses Groq LLM (Llama-3.1) for answering queries and FAISS for efficient similarity search.

🚀 Features

📁 Upload multiple PDF files

🔍 Extract and chunk text from PDFs

📊 Store embeddings in FAISS vector database

🤖 Query using Groq LLM (Llama-3.1-8b-instant)

🎨 Interactive Streamlit UI

⚡ Fast and accurate answers based only on your documents

🛠️ Tech Stack

Python 3.10+

Streamlit
 – UI framework

FAISS
 – Vector database

Groq API
 – LLM inference

PyPDF2
 – PDF text extraction

LangChain
 – Integration

📂 Project Structure
doc_ai/
│── data/               # Store your PDF files
│── storage/            # FAISS index saved here
│── src/
│   ├── ui.py           # Streamlit app (UI + backend)
│   ├── ingest.py       # (Optional) for preprocessing PDFs
│   ├── utils.py        # Utility functions
│── .env                # Environment variables (API keys)
│── requirements.txt    # Python dependencies
│── README.md           # Project documentation

⚙️ Setup & Installation
1. Clone the repository
git clone https://github.com/yourusername/multi-pdf-ai-assistant.git
cd multi-pdf-ai-assistant

2. Create virtual environment & install dependencies
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt

3. Set your API key

Create a .env file in the root folder:

GROQ_API_KEY=gsk_your_actual_api_key_here

4. Run the app
streamlit run src/ui.py

📖 Usage

Upload one or more PDF files from sidebar

Wait for text extraction & embeddings creation

Ask questions in the input box

Get answers based only on your PDFs

✅ Example Queries

"Summarize the main topics in these documents"

"What methodology was used in the research?"

"Who are the key authors?"

"What are the recommendations or conclusions?"

📌 TODO / Improvements

🔒 Add authentication for multi-user access

💾 Save and load FAISS index for faster reuse

🌐 Add support for web-based PDF ingestion

🖼️ Extract text from images using OCR (e.g., Tesseract)