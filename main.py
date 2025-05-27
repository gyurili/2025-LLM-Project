from src.utils.shared_cache import set_cache_dirs
set_cache_dirs()

import os
import time
from langsmith import trace
from dotenv import load_dotenv
from src.loader.loader_main import loader_main
from src.embedding.embedding_main import embedding_main
from src.embedding.vector_db import generate_embedding
from src.retrieval.retrieval_main import retrieval_main
from src.generator.generator_main import generator_main
from src.utils.config import load_config
from src.utils.path import get_project_root_dir
from src.generator.generator_main import load_chat_history


'''
    TODO:
    - 각자 main수정에 맞게 generator_main, retrieval_main, embedding_main, loader_main 수정
    - 임베딩, 모델인포, 컨피그, dotenv등은 전역적으로 한번만 선언
'''

project_root = get_project_root_dir()     
dotenv_path = os.path.join(project_root, ".env")
load_dotenv(dotenv_path=dotenv_path)
config = load_config(project_root)
embeddings = generate_embedding(config["embedding"]["embed_model"])
chat_history = load_chat_history(config)
# model_info = 

def rag_pipeline(config, model_info=None, is_save=True):
    try:
        with trace(name="rag_pipeline") as run:

            with trace(name="loader_main"):
                chunks = loader_main(config, embeddings, chat_history)
                
            with trace(name="embedding_main"):
                vector_store = embedding_main(config, chunks, embeddings=embeddings, is_save=is_save)

            with trace(name="retrieval_main"):
                docs = retrieval_main(config, vector_store, chunks, embeddings=embeddings)

            with trace(name="generator_main"):
                start_time = time.time()
                answer = generator_main(docs, config, model_info=model_info)
                end_time = time.time()
                elapsed = round(end_time - start_time, 2)

            run.add_outputs({
                "query": config["retriever"]["query"],
                "model_type": config["generator"]["model_type"],
                "model_name": config["generator"]["model_name"],
                "max_length": config["generator"]["max_length"],
                "num_chunks": len(chunks),
                "num_retrieved_docs": len(docs),
                "answer_length": len(answer),
                "final_answer": answer
            })
            
            return docs, answer, elapsed
            
    except Exception as e:
        print(f"❌ 로깅 에러: {e}")

if __name__ == "__main__":
    rag_pipeline(config)