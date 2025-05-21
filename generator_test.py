import os
from langsmith import trace
from dotenv import load_dotenv
from src.embedding.embedding_main import embedding_main
from src.retrieval.retrieval_main import retrieval_main
from src.loader.loader_main import loader_main
from src.generator.generator_main import generator_main
from src.utils.config import load_config
from src.utils.path import get_project_root_dir

def generator_test():
    with trace(name="generator_test") as run:
        project_root = get_project_root_dir()
        print(f"Project root directory: {project_root}")

        config_path = os.path.join(project_root, "config.yaml")
        print(f"Config file path: {config_path}")
        
        dotenv_path = os.path.join(project_root, ".env")
        load_dotenv(dotenv_path=dotenv_path)

        config = load_config(config_path)
        print("✅ Config 로드 완료")

        with trace(name="loader_main"):
            chunks = loader_main(config)
            print("✅ 데이터 로드 완료")

        with trace(name="embedding_main"):
            vector_store = embedding_main(config, chunks, is_save=True)
            print("✅ 벡터 DB 생성 완료")

        with trace(name="retrieval_main"):
            docs = retrieval_main(config, vector_store, chunks)
            print("✅ 문서 검색 완료")

        with trace(name="generator_main"):
            answer = generator_main(docs, config)
            print("✅ 답변 생성 완료")

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


if __name__ == "__main__":
    generator_test()