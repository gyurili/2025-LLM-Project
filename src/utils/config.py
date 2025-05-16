import os
import yaml


def load_config(config_path: str) -> dict:
    """
    YAML 파일에서 설정을 로드합니다.

    Args:
        config_path (str): YAML 파일 경로

    Returns:
        dict: 설정 딕셔너리
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ [FileNotFound] (config.load_config) 설정 파일을 찾을 수 없습니다: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def check_config(config: dict) -> None:
    """
    설정 딕셔너리의 유효성을 검사합니다.

    Args:
        config (dict): 설정 딕셔너리

    Raises:
        ValueError: 설정이 유효하지 않을 경우
    """
    if not isinstance(config, dict):
        raise ValueError("❌ [Type] (config.check_config) 설정은 딕셔너리여야 합니다.")
    
    # settings
    settings_config = config.get("settings", {})
    if not isinstance(settings_config, dict):
        raise ValueError("❌ [Type] (config.check_config.settings) 설정은 딕셔너리여야 합니다.")
    else:
        # verbose
        verbose = settings_config.get("verbose", False)
        if not isinstance(verbose, bool):
            raise ValueError("❌ [Type] (config.check_config.settings.verbose) verbose는 True 또는 False여야 합니다.")
        
    
    
    # data
    data_config = config.get("data", {})
    if not isinstance(data_config, dict):
        raise ValueError("❌ [Type] (config.check_config.data) 데이터 설정은 딕셔너리여야 합니다.")
    
    
    # embedding
    embedding_config = config.get("embedding", {})
    if not isinstance(embedding_config, dict):
        raise ValueError("❌ [Type] (config.check_config.embedding) 임베딩 설정은 딕셔너리여야 합니다.")
    

    # retriever
    retriever_config = config.get("retriever", {})
    if not isinstance(retriever_config, dict):
        raise ValueError("❌ [Type] (config.check_config.retriever) 검색기 설정은 딕셔너리여야 합니다.")
    

    # generator
    generator_config = config.get("generator", {})
    if not isinstance(generator_config, dict):
        raise ValueError("❌ [Type] (config.check_config.generator) 생성기 설정은 딕셔너리여야 합니다.")
    