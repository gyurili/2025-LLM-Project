import os
from langsmith import trace
from dotenv import load_dotenv
from src.loader.loader_main import loader_main
from src.utils.config import load_config
from src.utils.path import get_project_root_dir

def loader_test():
    try: 
        with trace(name="loader_test") as run:
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

            run.add_outputs({
                "top_k": config["data"]["top_k"],
                "file_type": config["data"]["file_type"],
                "apply_ocr": config["data"]["apply_ocr"],
                "splitter": config["data"]["splitter"],
                "chunk_size": config["data"]["chunk_size"],
                "chunk_overlap": config["data"]["chunk_overlap"],
                "num_chunks": len(chunks),
                "num_documents": len(set(doc.metadata["파일명"] for doc in chunks)),
            })
    except Exception as e:
        print(f"❌ 로깅 에러: {e}")

if __name__ == "__main__":
    loader_test()