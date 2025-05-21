import argparse

if __name__ == '__main__':
    import os

    from src.utils.path import get_project_root_dir
    from src.utils.config import load_config
    from src.loader.loader_main import loader_main
    from src.embedding.embedding_main import embedding_main

    parser = argparse.ArgumentParser(description="parser 엔트리 포인트")
    parser.add_argument("--is_save", action="store_true", help="저장하기 모드")
    args = parser.parse_args()
    
    project_root = get_project_root_dir()
    print(f"Project root directory: {project_root}")
    config_path = os.path.join(project_root, "config.yaml")
    print(f"Config file path: {config_path}")

    config = load_config(config_path)
    print("✅ Config 로드 완료")
    
    if args.is_save:
        chunks = loader_main(config)
    else: 
        chunks = []
        
    vector_store = embedding_main(config, chunks, is_save=args.is_save)
    print("✅ 벡터 DB 생성 완료")
