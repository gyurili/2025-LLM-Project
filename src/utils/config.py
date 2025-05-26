import os
import yaml


def check_config(config: dict) -> None:
    """
    설정 딕셔너리의 유효성을 검사합니다.

    Args:
        config (dict): 설정 딕셔너리

    Raises:
        ValueError: 설정이 유효하지 않을 경우
    """
    if not isinstance(config, dict):
        raise ValueError("❌ (config.check_config) 설정은 딕셔너리여야 합니다.")
    
    # settings
    settings_config = config.get("settings", {})
    if not isinstance(settings_config, dict):
        raise ValueError("❌ (config.check_config.settings) 설정은 딕셔너리여야 합니다.")
    else:
        # verbose
        verbose = settings_config.get("verbose", False)
        if not isinstance(verbose, bool):
            raise ValueError("❌ (config.check_config.settings.verbose) verbose는 True 또는 False여야 합니다.")
        project_root = settings_config.get("project_root", "2025-LLM-Project")
        if not isinstance(project_root, str):
            raise ValueError("❌ (config.check_config.settings.project_root) project_root는 문자열이어야 합니다.")

    # data
    data_config = config.get("data", {})
    if not isinstance(data_config, dict):
        raise ValueError("❌ (config.check_config.data) 데이터 설정은 딕셔너리여야 합니다.")
    else:
        # load
        folder_path = data_config.get("folder_path", "data/files")
        if not isinstance(folder_path, str):
            raise ValueError("❌ (config.check_config.data.folder_path) 폴더 경로는 문자열이어야 합니다.")
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"❌ (config.check_config.data.folder_path) 폴더 경로가 존재하지 않습니다: {folder_path}")
        
        data_list_path = data_config.get("data_list_path", "data/data_list.csv")
        if not isinstance(data_list_path, str):
            raise ValueError("❌ (config.check_config.data.data_list_path) 데이터 목록 경로는 문자열이어야 합니다.")
        if not os.path.exists(data_list_path):
            raise FileNotFoundError(f"❌ (config.check_config.data.data_list_path) 파일 목록 경로가 존재하지 않습니다: {data_list_path}")
        
        top_k = data_config.get("top_k", 5)
        if not isinstance(top_k, int):
            raise ValueError("❌ (config.check_config.data.top_k) top_k는 정수여야 합니다.")
        if top_k < 1:
            print("⚠️ (config.check_config.data.top_k) top_k는 1이상의 정수여야 합니다.")
        if top_k > 100:
            print("⚠️ (config.check_config.data.top_k) top_k는 100이하의 정수여야 합니다.")

        file_type = data_config.get("file_type", "all")
        if file_type not in ["hwp", "pdf", "all"]:
            raise ValueError("❌ (config.check_config.data.file_type) file_type은 'hwp', 'pdf', 'all' 중 하나여야 합니다.")
        
        apply_ocr = data_config.get("apply_ocr", False)
        if not isinstance(apply_ocr, bool):
            raise ValueError("❌ (config.check_config.data.apply_ocr) apply_ocr는 True 또는 False여야 합니다.")
        
        # chunk
        splitter = data_config.get("splitter", "section")
        if not isinstance(splitter, str):
            raise ValueError("❌ (config.check_config.data.splitter) 청크 분할기 타입은 문자열이어야 합니다.")
        if splitter not in ["recursive", "token", "section"]:
            raise ValueError("❌ (config.check_config.data.splitter) splitter는 'recursive', 'token', 'section' 중 하나여야 합니다.")
        chunk_size = data_config.get("chunk_size", 1000)
        if chunk_size < 1 or not isinstance(chunk_size, int):
            raise ValueError("❌ (config.check_config.data.chunk_size) 청크 크기는 1 이상의 정수여야 합니다.")
        chunk_overlap = data_config.get("chunk_overlap", 250)
        if chunk_overlap < 0 or not isinstance(chunk_overlap, int):
            raise ValueError("❌ (config.check_config.data.chunk_overlap) 청크 오버랩은 0 이상의 정수여야 합니다.")
    
    
    # embedding
    embedding_config = config.get("embedding", {})
    if not isinstance(embedding_config, dict):
        raise ValueError("❌ (config.check_config.embedding) 임베딩 설정은 딕셔너리여야 합니다.")
    else:
        embed_mode = embedding_config.get("embed_model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        if not isinstance(embed_mode, str):
            raise ValueError("❌ (config.check_config.embedding.embed_model) 임베딩 모델 이름은 문자열이어야 합니다.")

        db_type = embedding_config.get("db_type", "faiss")
        if db_type not in ["faiss", "chroma"]:
            raise ValueError("❌ (config.check_config.embedding.db_type) 지원하지 않는 벡터 DB 타입입니다. ('faiss' 또는 'chroma' 사용)")
        
        vector_db_path = embedding_config.get("vector_db_path", "data/vector_db")
        if not isinstance(vector_db_path, str):
            raise ValueError("❌ (config.check_config.embedding.vector_db_path) 벡터 DB 경로는 문자열이어야 합니다.")
        if not os.path.exists(vector_db_path):
            raise FileNotFoundError(f"❌ (config.check_config.embedding.vector_db_path) 벡터 DB 경로가 존재하지 않습니다: {vector_db_path}")


    # retriever
    retriever_config = config.get("retriever", {})
    if not isinstance(retriever_config, dict):
        raise ValueError("❌ (config.check_config.retriever) 검색기 설정은 딕셔너리여야 합니다.")
    else:
        query = retriever_config.get("query", "")
        if not isinstance(query, str):
            raise ValueError("❌ (config.check_config.retriever.query) 쿼리는 문자열이어야 합니다.")
        
        search_type = retriever_config.get("search_type", "hybrid")
        if search_type not in ["similarity", "hybrid"]:
            raise ValueError("❌ (config.check_config.retriever.search_type) search_type은 'similarity' 또는 'hybrid' 중 하나여야 합니다.")

        top_k = retriever_config.get("top_k", 10)
        if not isinstance(top_k, int):
            raise ValueError("❌ (config.check_config.retriever.top_k) top_k는 정수여야 합니다.")
        if top_k < 1:
            print("⚠️ (config.check_config.retriever.top_k) top_k는 1이상의 정수여야 합니다.")
        
        rerank = retriever_config.get("rerank", True)
        if not isinstance(rerank, bool):
            raise ValueError("❌ (config.check_config.retriever.rerank) rerank는 True 또는 False여야 합니다.")

        rerank_top_k = retriever_config.get("rerank_top_k", 5)
        if not isinstance(rerank_top_k, int):
            raise ValueError("❌ (config.check_config.retriever.rerank_top_k) rerank_top_k는 정수여야 합니다.")
        if rerank_top_k < 1:
            print("⚠️ (config.check_config.retriever.rerank_top_k) rerank_top_k는 1 이상의 정수여야 합니다.")


    # generator
    generator_config = config.get("generator", {})
    if not isinstance(generator_config, dict):
        raise ValueError("❌ (config.check_config.generator) 생성기 설정은 딕셔너리여야 합니다.")
    else:
        model_type = generator_config.get("model_type", "huggingface")
        if model_type not in ["huggingface", "openai"]:
            raise ValueError("❌ (config.check_config.generator.model_type) 지원하지 않는 모델 타입입니다. ('huggingface' 또는 'openai' 사용)")
        
        model_name = generator_config.get("model_name", "")
        if not isinstance(model_name, str):
            raise ValueError("❌ (config.check_config.generator.model_name) 모델 이름은 문자열이어야 합니다.")
        
        max_length = generator_config.get("max_length", 512)
        if not isinstance(max_length, int):
            raise ValueError("❌ (config.check_config.generator.max_length) 최대 길이는 정수여야 합니다.")
        if max_length < 1:
            raise ValueError("❌ (config.check_config.generator.max_length) 최대 길이는 1이상의 정수여야 합니다.")
        
        use_quantization = generator_config.get("use_quantization", True)
        if not isinstance(use_quantization, bool):
            raise ValueError("❌ (config.check_config.generator.use_quantization) 양자화는 True/ False여야 합니다.")
    
    # chat_history
    chat_history = config.get("chat_history")
    if not isinstance(chat_history, list):
        raise ValueError("❌ (config.check_config.chat_history) chat_history는 list이어야 합니다.")


def load_config(project_root: str) -> dict:
    """
    YAML 파일에서 설정을 로드합니다.

    Args:
        project_root (str): 프로젝트 루트

    Returns:
        dict: 설정 딕셔너리
    """
    config_path = os.path.join(project_root, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ (config.load_config) 설정 파일을 찾을 수 없습니다: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if "settings" not in config or not isinstance(config["settings"], dict):
        config["settings"] = {}
    config["settings"]["project_root"] = project_root

    try:
        # 설정 유효성 검사
        check_config(config)

        # verbose 모드일 경우 전체 설정 출력
        if config.get("settings", {}).get("verbose", False):
            print("\n📄 [Verbose] 최종 설정 내용:")
            print(yaml.dump(config, allow_unicode=True, sort_keys=False))
    
    # 예외 처리
    except (FileNotFoundError, PermissionError) as e:
        print(f"❌ [File] 파일 접근 오류:\n  {e}")

    except yaml.YAMLError as e:
        print(f"❌ [YAML] 설정 파일 파싱 오류:\n  {e}")

    except (ValueError, TypeError) as e:
        print(f"❌ [Config] 설정값 오류:\n  {e}")

    except Exception as e:
        print(f"❌ [Unexpected] 예상치 못한 오류 발생:\n  {e}")

    return config