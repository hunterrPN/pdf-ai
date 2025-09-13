import faiss, pickle, numpy as np
from sentence_transformers import SentenceTransformer
from src.llm import get_groq_llm

STORAGE_DIR = "storage/"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

class DocQASystem:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.index = faiss.read_index(STORAGE_DIR + "faiss.index")
        with open(STORAGE_DIR + "metadata.pkl", "rb") as f:
            data = pickle.load(f)
        self.texts, self.metadata = data["texts"], data["metadata"]
        self.llm = get_groq_llm()

    def query(self, user_query, top_k=3):
        q_emb = self.model.encode([user_query], convert_to_numpy=True)
        D, I = self.index.search(q_emb, top_k)
        retrieved_chunks = [self.texts[i] for i in I[0]]
        sources = [self.metadata[i] for i in I[0]]

        context = "\n\n".join(retrieved_chunks)
        prompt = f"""You are a helpful research assistant.
Answer the query based on the following context:

{context}

Query: {user_query}
Answer:"""

        response = self.llm.invoke(prompt)
        return {"answer": response.content, "sources": sources}
