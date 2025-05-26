
import os
import time
from langsmith import trace
from dotenv import load_dotenv
from src.loader.loader_main import loader_main
from src.embedding.embedding_main import embedding_main
from src.retrieval.retrieval_main import retrieval_main
from src.generator.generator_main import generator_main
from src.utils.config import load_config
from src.utils.path import get_project_root_dir
from src.utils.shared_cache import set_cache_dirs
set_cache_dirs()

def rag_pipeline():
    try:
        with trace(name="rag_pipeline") as run:
            project_root = get_project_root_dir()
            print(f"Project root directory: {project_root}")

            config_path = os.path.join(project_root, "config.yaml")
            print(f"Config file path: {config_path}")
            
            dotenv_path = os.path.join(project_root, ".env")
            load_dotenv(dotenv_path=dotenv_path)

            config = load_config(project_root)

            with trace(name="loader_main"):
                chunks = loader_main(config)

            with trace(name="embedding_main"):
                vector_store = embedding_main(config, chunks, is_save=False)

            with trace(name="retrieval_main"):
                docs = retrieval_main(config, vector_store, chunks)

            start_time = time.time()
            with trace(name="generator_main"):
                answer = generator_main(docs, config)
                print("✅ 답변 생성 완료")
            end_time = time.time()
            elapsed = round(end_time - start_time, 2)
            
            run.add_outputs({
                "num_chunks": len(chunks),
                "num_retrieved_docs": len(docs),
                "final_answer": answer
            })

        return {
            "docs": docs,
            "answer": answer,
            "elapsed_time": elapsed
        }
    except Exception as e:
        print(f"❌ 로깅 에러: {e}")

if __name__ == "__main__":
    rag_pipeline()