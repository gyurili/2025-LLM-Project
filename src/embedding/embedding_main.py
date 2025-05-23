import os
import shutil
from typing import List, Union
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma

from src.embedding.vector_db import generate_vector_db, load_vector_db
from src.utils.path import get_project_root_dir

def generate_index_name(config: dict) -> str:
    """
    config 설정값을 조합하여 벡터 DB 인덱스 이름을 생성합니다.

    구성 요소:
    - data.type
    - chunk.splitter
    - embedding.model
    - embedding.db_type

    모델명이 경로 형태일 경우 마지막 항목만 사용하며,
    하이픈(-), 슬래시(/), 공백은 언더스코어(_)로 변환합니다.

    Args:
        config (dict): 설정 딕셔너리

    Returns:
        str: 자동 생성된 인덱스 이름 (예: all_100_recursive_openai_faiss_index)
    """
    data_type = config.get("data", {}).get("file_type", "all")
    splitter = config.get("data", {}).get("splitter", "recursive")
    model = config.get("embedding", {}).get("embed_model", "default")
    db_type = config.get("embedding", {}).get("db_type", "faiss")

    # 모델 이름에서 마지막 슬래시 기준 요소만 추출 후 특수문자 제거
    model_key = model.split("/")[-1] if "/" in model else model
    model_key = model_key.replace('-', '_').replace(' ', '_')
    if config['data']['top_k'] == 100:
        return f"{data_type}_{config['data']['top_k']}_{splitter}_{model_key}_{db_type}"
    else:
        return f"{data_type}_{splitter}_{model_key}_{db_type}"


def embedding_main(config: dict, chunks: List[Document], is_save:bool = False) -> Union[FAISS, Chroma]:
    """
    벡터 DB를 생성합니다.

    Args:
        config (dict): 설정 정보
        chunks (List[Document]): 청크 리스트(loader_main에서 생성된 Document 객체 리스트)

    Returns:
        VectorStore 객체
    """
    if not isinstance(chunks, list) or not all(isinstance(chunk, Document) for chunk in chunks):
        raise ValueError("❌(embedding.embedding_main) chunks는 Document 객체의 리스트여야 합니다.")

    embed_config = config.get("embedding", {})
    embed_model = embed_config.get("embed_model", "openai")
    project_root = get_project_root_dir()
    vector_db_path = os.path.join(project_root, embed_config.get("vector_db_path", "data"))
    if not os.path.exists(vector_db_path):
        os.makedirs(vector_db_path, exist_ok=True)
    db_type = embed_config.get("db_type", "faiss")

    if db_type == "faiss":
        faiss_file = os.path.join(vector_db_path, f"{generate_index_name(config)}.faiss")
        pkl_file = os.path.join(vector_db_path, f"{generate_index_name(config)}.pkl")
        db_exists = os.path.exists(faiss_file) and os.path.exists(pkl_file)
    elif db_type == "chroma":
        chroma_dir = os.path.join(vector_db_path, generate_index_name(config))
        sqlite_path = os.path.join(chroma_dir, "chroma.sqlite3")

        # 필수 요소가 모두 존재하는지 확인
        has_sqlite = os.path.exists(sqlite_path)
        has_index_dirs = any(
            os.path.isdir(os.path.join(chroma_dir, d))
            and len(os.listdir(os.path.join(chroma_dir, d))) >= 4
            for d in os.listdir(chroma_dir)
            if os.path.isdir(os.path.join(chroma_dir, d))
        ) if os.path.exists(chroma_dir) else False
    
        db_exists = has_sqlite and has_index_dirs
    
        # 불완전한 DB면 삭제
        if os.path.exists(chroma_dir) and not db_exists:
            print("⚠️ 불완전한 Chroma 벡터 DB가 감지되어 삭제합니다.")
            shutil.rmtree(chroma_dir)
            db_exists = False
        db_exists = os.path.isdir(chroma_dir) and os.path.exists(os.path.join(chroma_dir, "chroma.sqlite3"))
    else:
        raise ValueError(f"❌(embedding.embedding_main) 지원하지 않는 DB 타입입니다: {db_type}")

    if is_save:
        # Save모드일 때 자동 생성
        vector_store = generate_vector_db(chunks, embed_model, generate_index_name(config), db_type)
        print("✅ Vector DB 생성 완료")
    else:
        # Save모드가 아닐 때 불러오기 
        vector_store = load_vector_db(vector_db_path, embed_model, generate_index_name(config), db_type)
        print("✅ Vector DB 로드 완료")

    
    return vector_store