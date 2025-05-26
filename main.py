from src.utils.shared_cache import set_cache_dirs
set_cache_dirs()

import os
from langsmith import trace
from dotenv import load_dotenv
from src.loader.loader_main import loader_main
from src.embedding.embedding_main import embedding_main
from src.embedding.vector_db import generate_embedding
from src.retrieval.retrieval_main import retrieval_main
from src.generator.generator_main import generator_main
from src.utils.config import load_config
from src.utils.path import get_project_root_dir


'''
    TODO:
    - 각자 main수정에 맞게 generator_main, retrieval_main, embedding_main, loader_main 수정
    - 임베딩, 모델인포, 컨피그, dotenv등은 전역적으로 한번만 선언
'''

project_root = get_project_root_dir()     
dotenv_path = os.path.join(project_root, ".env")
load_dotenv(dotenv_path=dotenv_path)
config = load_config(project_root)

embedder = generate_embedding(config["embedding"]["embed_model"])

def rag_pipeline(config, embedder, model_info=None):
    try:
        with trace(name="rag_pipeline") as run:

            with trace(name="loader_main"):
                chunks = loader_main(config, embedder=embedder)

            with trace(name="embedding_main"):
                vector_store = embedding_main(config, chunks, is_save=False)

            with trace(name="retrieval_main"):
                docs = retrieval_main(config, vector_store, chunks)

            with trace(name="generator_main"):
                answer = generator_main(docs, config)

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

    except Exception as e:
        print(f"❌ 로깅 에러: {e}")

    return docs, answer, 

if __name__ == "__main__":
    rag_pipeline()