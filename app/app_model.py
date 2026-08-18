from pydantic import BaseModel

class QueryRequest(BaseModel):
    question: str
    history: list[dict] = []

class QueryResponse(BaseModel):
    answer: str
   # sources: list[str]