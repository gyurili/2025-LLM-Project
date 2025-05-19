if __name__ == '__main__':
    import os

    from src.utils.path import get_project_root_dir
    from src.utils.config import load_config
    from src.loader.loader_main import loader_main


    project_root = get_project_root_dir()
    print(f"Project root directory: {project_root}")
    config_path = os.path.join(project_root, "config.yaml")
    print(f"Config file path: {config_path}")

    config = load_config(config_path)
    print("✅ Config 로드 완료")

    chunks = loader_main(config)

    print("✅ 데이터 로드 및 전처리 완료")