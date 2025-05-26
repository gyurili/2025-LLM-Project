from src.utils.shared_cache import set_cache_dirs
set_cache_dirs()

import os
from langsmith import trace
from dotenv import load_dotenv

from src.embedding.vector_db import generate_embedding
from src.embedding.embedding_main import embedding_main
from src.loader.loader_main import loader_main
from src.utils.config import load_config
from src.utils.path import get_project_root_dir

def embedding_test():
    try:
        with trace(name="embedding_test") as run:
            project_root = get_project_root_dir()

            config_path = os.path.join(project_root, "config.yaml")
            dotenv_path = os.path.join(project_root, ".env")
            load_dotenv(dotenv_path=dotenv_path)

            config = load_config(project_root)

            with trace(name="loader_main"):
                chunks = loader_main(config)

            with trace(name="embedding_main"):
                embed_model_name = config["embedding"]["embed_model"]
                embeddings = generate_embedding(embed_model_name)
                vector_store = embedding_main(config, chunks, embeddings, is_save=True)

            run.add_outputs({
                "embed_model": embed_model_name,
                "db_type": config["embedding"]["db_type"],
                "vector_db_path": config["embedding"]["vector_db_path"],
                "top_k": config["data"]["top_k"],
                "splitter": config["data"]["splitter"],
                "chunk_size": config["data"]["chunk_size"],
                "chunk_overlap": config["data"]["chunk_overlap"],
                "num_chunks": len(chunks),
                "is_save": True,
            })
    except Exception as e:
        print(f"❌ 로깅 에러: {e}")

if __name__ == "__main__":
    embedding_test()