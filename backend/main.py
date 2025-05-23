from fastapi import FastAPI
from pydantic import BaseModel
from .rag_api import run_rag_pipeline

app = FastAPI()                             # FastAPI 앱 생성

class QueryRequest(BaseModel):
    query: str                              # 쿼리는 반드시 문자열로 받음

@app.post("/rag")                           # /rag 경로로 POST 요청이 오면 실행할 함수
def handle_rag(request: QueryRequest): 
    return run_rag_pipeline(request.query)