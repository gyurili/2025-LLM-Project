if __name__ == "__main__":
    import os
    from src.embedding.embedding_main import embedding_main
    from src.retrieval.retrieval_main import retrieval_main
    from src.loader.loader_main import loader_main
    from src.generator.generator_main import generator_main
    from src.utils.config import load_config
    from src.utils.path import get_project_root_dir
    from dotenv import load_dotenv
    
    project_root = get_project_root_dir()
    print(f"Project root directory: {project_root}")
    config_path = os.path.join(project_root, "config.yaml")
    print(f"Config file path: {config_path}")

    config = load_config(config_path)

    # .env 불러오기
    dotenv_path = os.path.join(project_root, ".env")
    load_dotenv(dotenv_path=dotenv_path)

    # 데이터 로드 및 청킹
    chunk = loader_main(config)

    # 벡터 DB 생성
    vector_store = embedding_main(config, chunk)

    # 유사도 검색
    docs = retrieval_main(config, vector_store, chunk)
    print("✅ 검색 완료")

    # 답변 생성
    answer = generator_main(docs, config)