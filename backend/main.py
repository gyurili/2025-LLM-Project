from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from copy import deepcopy
from src.generator.chat_history import load_chat_history
from src.utils.config import load_config
from src.utils.path import get_project_root_dir
from src.embedding.vector_db import generate_embedding
from main import get_generation_model, rag_pipeline
import os
from dotenv import load_dotenv

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 입력 데이터 형식
class QueryRequest(BaseModel):
    query: str
    chat_history: list
    session_id: Optional[str] = None
    config: dict

class QueryResponse(BaseModel):
    answer: str
    elapsed: float
    docs: list

# 기본 설정
project_root = get_project_root_dir()
config = load_config(project_root)
dotenv_path = os.path.join(project_root, ".env")
load_dotenv(dotenv_path=dotenv_path)

embed_model_name = config["embedding"]["embed_model"]
embeddings = generate_embedding(embed_model_name)

model_type = config["generator"]["model_type"]
model_name = config["generator"]["model_name"]
use_quantization = config["generator"]["use_quantization"]
model_info = get_generation_model(model_type, model_name, use_quantization)

# API 호출
from copy import deepcopy

@app.post("/chat", response_model=QueryResponse)
def chat(request: QueryRequest):
    local_config = deepcopy(request.config)
    local_config["chat_history"] = request.chat_history

    if request.chat_history:
        chat_summary = load_chat_history(local_config, model_info)
        local_config["retriever"]["query"] = f"이전 질문 요약: {chat_summary}\n질문: {request.query}"
    else:
        local_config["retriever"]["query"] = request.query

    docs, answer, elapsed = rag_pipeline(local_config, embeddings, request.chat_history, model_info, is_save=True, session_id=request.session_id)

    docs_result = [
        {
            "metadata": doc.metadata,
            "content": doc.page_content
        } for doc in docs
    ]

    return QueryResponse(answer=answer, elapsed=elapsed, docs=docs_result)
