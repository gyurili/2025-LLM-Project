if __name__ == "__main__":
    import os
    from src.embedding.embedding_main import embedding_main
    from src.retrieval.retrieval_main import retrieval_main
    from src.loader.loader_main import loader_main
    from src.generator.generator_main import generator_main, generate_with_clarification
    from src.utils.config import load_config
    from src.utils.path import get_project_root_dir
    from dotenv import load_dotenv
  
    project_root = get_project_root_dir()
    print(f"Project root directory: {project_root}")
    config_path = os.path.join(project_root, "config.yaml")
    print(f"Config file path: {config_path}")
    
    dotenv_path = os.path.join(project_root, ".env")
    load_dotenv(dotenv_path=dotenv_path)

    config = load_config(config_path)
    print("✅ Config 로드 완료")

    chunks = loader_main(config)

    vector_store = embedding_main(config, chunks, is_save=False)
    print("✅ 벡터 DB 생성 완료")

    docs = retrieval_main(config, vector_store, chunks)
    print("✅ retrieval 검색 완료")

    answer = generate_with_clarification(docs, config)
    print("✅ 답변 생성 완료")