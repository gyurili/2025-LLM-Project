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
        if verbose:
            print("    -Verbose 모드로 실행 중입니다.")
        
    
    
    # data
    data_config = config.get("data", {})
    if not isinstance(data_config, dict):
        raise ValueError("❌ [Type] (config.check_config.data) 데이터 설정은 딕셔너리여야 합니다.")
    else:
        # load
        folder_path = data_config.get("folder_path", "data/files"):
        if not isinstance(folder_path, str):
            raise ValueError("❌ [Type] (config.check_config.data.folder_path) 폴더 경로는 문자열이어야 합니다.")
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"❌ [FileNotFound] (config.check_config.data.folder_path) 폴더 경로가 존재하지 않습니다: {folder_path}")
        
        data_list_path = data_config.get("data_list_path", "data/data_list.csv")
        if not isinstance(data_list_path, str):
            raise ValueError("❌ [Type] (config.check_config.data.data_list_path) 데이터 목록 경로는 문자열이어야 합니다.")
        if not os.path.exists(data_list_path):
            raise FileNotFoundError(f"❌ [FileNotFound] (config.check_config.data.data_list_path) 파일 목록 경로가 존재하지 않습니다: {data_list_path}")
        
        limit = data_config.get("limit", 5)
        if not isinstance(limit, int):
            raise ValueError("❌ [Type] (config.check_config.data.limit) limit은 정수여야 합니다.")
        if limit < 1:
            limit = 1
            print("⚠️ [Warning] (config.check_config.data.limit) limit은 0보다 큰 정수여야 합니다. 최소값 1로 설정합니다.")
        if limit > 100:
            limit = 100
            print("⚠️ [Warning] (config.check_config.data.limit) limit은 100보다 작거나 같아야 합니다. 최대값 100으로 설정합니다.")

        file_type = data_config.get("file_type", "all")
        if file_type not in ["hwp", "pdf", "all"]:
            raise ValueError("❌ [Value] (config.check_config.data.file_type) file_type은 'hwp', 'pdf', 'all' 중 하나여야 합니다.")
        
        apply_ocr = data_config.get("apply_ocr", False)
        if not isinstance(apply_ocr, bool):
            raise ValueError("❌ [Type] (config.check_config.data.apply_ocr) apply_ocr는 True 또는 False여야 합니다.")
        
        # chunk
        spliiter = data_config.get("splitter", "recursive")
        if not isinstance(spliiter, str):
            raise ValueError("❌ [Type] (config.check_config.data.splitter) 청크 분할기 타입은 문자열이어야 합니다.")
        chunk_size = data_config.get("size", 300)
        if chunk_size < 1 or not isinstance(chunk_size, int):
            raise ValueError("❌ [Type] (config.check_config.data.size) 청크 크기는 1 이상의 정수여야 합니다.")
        chunk_overlap = data_config.get("overlap", 50)
        if chunk_overlap < 0 or not isinstance(chunk_overlap, int):
            raise ValueError("❌ [Type] (config.check_config.data.overlap) 청크 오버랩은 0 이상의 정수여야 합니다.")
    
    
    # embedding
    embedding_config = config.get("embedding", {})
    if not isinstance(embedding_config, dict):
        raise ValueError("❌ [Type] (config.check_config.embedding) 임베딩 설정은 딕셔너리여야 합니다.")
    else:
        embed_mode = embedding_config.get("embed_model_name", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        if not isinstance(embed_mode, str):
            raise ValueError("❌ [Type] (config.check_config.embedding.embed_model_name) 임베딩 모델 이름은 문자열이어야 합니다.")

        db_type = embedding_config.get("db_type", "faiss")
        if db_type not in ["faiss", "chroma"]:
            raise ValueError("❌ [Value] (config.check_config.embedding.db_type) 지원하지 않는 벡터 DB 타입입니다. ('faiss' 또는 'chroma' 사용)")
        
        vector_db_path = embedding_config.get("vector_db_path", "data")
        if not isinstance(vector_db_path, str):
            raise ValueError("❌ [Type] (config.check_config.embedding.vector_db_path) 벡터 DB 경로는 문자열이어야 합니다.")
        if not os.path.exists(vector_db_path):
            raise FileNotFoundError(f"❌ [FileNotFound] (config.check_config.embedding.vector_db_path) 벡터 DB 경로가 존재하지 않습니다: {vector_db_path}")
    

    # retriever
    retriever_config = config.get("retriever", {})
    if not isinstance(retriever_config, dict):
        raise ValueError("❌ [Type] (config.check_config.retriever) 검색기 설정은 딕셔너리여야 합니다.")
    else:
        query = retriever_config.get("query", "")
        if not isinstance(query, str):
            raise ValueError("❌ [Type] (config.check_config.retriever.query) 쿼리는 문자열이어야 합니다.")
        
        top_k = retriever_config.get("top_k", 5)
        if not isinstance(top_k, int):
            raise ValueError("❌ [Type] (config.check_config.retriever.top_k) top_k는 정수여야 합니다.")
        if top_k < 1:
            top_k = 1
            print("⚠️ [Warning] (config.check_config.retriever.top_k) top_k는 0보다 큰 정수여야 합니다. 최소값 1로 설정합니다.")
        

    # generator
    generator_config = config.get("generator", {})
    if not isinstance(generator_config, dict):
        raise ValueError("❌ [Type] (config.check_config.generator) 생성기 설정은 딕셔너리여야 합니다.")
    else:
        model_type = generator_config.get("model_type", "huggingface")
        if model_type not in ["huggingface", "openai"]:
            raise ValueError("❌ [Value] (config.check_config.generator.model_type) 지원하지 않는 모델 타입입니다. ('huggingface' 또는 'openai' 사용)")
        
        model_name = generator_config.get("model_name", "")
        if not isinstance(model_name, str):
            raise ValueError("❌ [Type] (config.check_config.generator.model_name) 모델 이름은 문자열이어야 합니다.")
        
        max_length = generator_config.get("max_length", 512)
        if not isinstance(max_length, int):
            raise ValueError("❌ [Type] (config.check_config.generator.max_length) 최대 길이는 정수여야 합니다.")
        if max_length < 1:
            raise ValueError("❌ [Value] (config.check_config.generator.max_length) 최대 길이는 1보다 커야 합니다.")
        
    