from src.utils.shared_cache import set_cache_dirs
set_cache_dirs()
import os
from dotenv import load_dotenv
from langsmith import trace
from src.utils.path import get_project_root_dir
from src.utils.config import load_config
from src.loader.loader_main import loader_main
from src.embedding.embedding_main import embedding_main
from src.retrieval.retrieval_main import retrieval_main
from src.generator.generator_main import generator_main

def run_rag_pipeline(user_query: str) -> dict:
    try:
        with trace(name="rag_pipeline") as run:
            project_root = get_project_root_dir()
            dotenv_path = os.path.join(project_root, ".env")
            load_dotenv(dotenv_path=dotenv_path)

            config_path = os.path.join(project_root, "config.yaml")
            config = load_config(config_path)

            # 사용자 query 입력 받기
            config["retriever"]["query"] = user_query

            # 전체 파이프라인 실행
            chunks = loader_main(config)
            vector_store = embedding_main(config, chunks, is_save=True)
            docs = retrieval_main(config, vector_store, chunks)
            answer = generator_main(docs, config)

            return {
                "answer": answer,
                "num_chunks": len(chunks),
                "num_retrieved_docs": len(docs),
                "docs_preview": [doc.page_content[:200] for doc in docs]
            }
    except Exception as e:
        return {"error": str(e)}
