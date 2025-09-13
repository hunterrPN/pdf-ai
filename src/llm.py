from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

def get_groq_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",  # Groq model
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )
