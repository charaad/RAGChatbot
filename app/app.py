import time
from fastapi import FastAPI, HTTPException
from app.app_model import QueryRequest, QueryResponse
from main import rag_run


app = FastAPI(title="RAG Chatbot API")

@app.get("/")
def home():
    return {"message": "RAG API is running"}

@app.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    try:
        answer = rag_run(request.question, request.history)
        return QueryResponse(answer=answer)
    
    except Exception as e:
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            raise HTTPException(
                status_code=429,
                detail="API rate limit reached. Please wait a few seconds and try again.",
            )
        raise HTTPException(status_code=500, detail=str(e))