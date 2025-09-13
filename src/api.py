from fastapi import FastAPI
from src.query import DocQASystem

app = FastAPI()
qa_system = DocQASystem()

@app.get("/query")
def query_docs(q: str):
    result = qa_system.query(q)
    return result
