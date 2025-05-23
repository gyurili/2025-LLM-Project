import os
from typing import List, Union, Optional, Literal

import faiss
from dotenv import load_dotenv
from tqdm import tqdm

from langsmith import trace
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.base import VectorStore

from src.utils.path import get_project_root_dir


def generate_embedding(embed_model_name: str) -> Union[OpenAIEmbeddings, HuggingFaceEmbeddings]:
    """
    주어진 모델 이름에 따라 OpenAI 또는 HuggingFace 기반 임베딩 모델을 초기화합니다.

    Args:
        embed_model_name (str): 사용할 임베딩 모델 이름. 'openai' 또는 Hugging Face 모델 이름

    Returns:
        Union[OpenAIEmbeddings, HuggingFaceEmbeddings]: 초기화된 임베딩 모델 객체

    Raises:
        ValueError: API 키 누락 또는 모델 초기화 실패 시 예외 발생
    """
    try:
        if embed_model_name == "openai":
            load_dotenv()
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("❌ OPENAI_API_KEY가 .env에 정의되어 있지 않습니다.")
            return OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=api_key)
        else:
            return HuggingFaceEmbeddings(model_name=embed_model_name)
    except Exception as e:
        raise ValueError(f"❌ [Value] (vector_db.generate_embedding) 임베딩 모델 초기화 실패: {e}")


@trace(name="generate_vector_db")
def generate_vector_db(
    all_chunks: List[Document],
    embed_model_name: str,
    index_name: str,
    db_type: str = "faiss",
    is_save: bool = True,
    batch_size: int = 128
) -> Union[FAISS, Chroma]:
    """
    문서 청크 리스트를 이용해 FAISS 또는 Chroma 기반의 벡터 데이터베이스를 생성하고,
    지정된 경로에 저장할 수 있습니다.

    Args:
        all_chunks (List[Document]): 임베딩할 문서 청크 리스트
        embed_model_name (str): 사용할 임베딩 모델 이름
        index_name (str): 저장할 벡터 DB의 인덱스 이름
        db_type (str): 사용할 벡터 저장소 유형 ('faiss' 또는 'chroma')
        is_save (bool): DB 저장 여부 (faiss만 해당)
        batch_size (int): 문서 삽입 시 배치 크기

    Returns:
        Union[FAISS, Chroma]: 생성된 벡터 DB 인스턴스

    Raises:
        ValueError: 잘못된 입력 값이나 지원하지 않는 DB 타입 입력 시
        RuntimeError: 벡터 DB 생성 중 오류 발생 시
    """
    if not all_chunks or not isinstance(all_chunks, list):
        raise ValueError("❌ [Value] (vector_db.generate_vector_db) all_chunks는 비어 있지 않은 Document 리스트여야 합니다.")

    embeddings = generate_embedding(embed_model_name)
    print(f"📌 [Info] Embedding model: {embeddings.__class__.__name__}")

    try:
        if isinstance(embeddings, (HuggingFaceEmbeddings, OpenAIEmbeddings)):
            dimension = len(embeddings.embed_query("hello world"))
        else:
            dimension = 1536  # default
    except Exception as e:
        raise ValueError(f"❌ [Value] (vector_db.generate_vector_db) 임베딩 차원 계산 실패: {e}")

    try:
        output_path = os.path.join(get_project_root_dir(), "data")
        os.makedirs(output_path, exist_ok=True)

        if db_type == "faiss":
            print(f"📌 [info] Vector DB type: {db_type}")
            vector_store = FAISS(
                embedding_function=embeddings,
                index=faiss.IndexFlatL2(dimension),
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            )
            vector_store = add_docs_in_batch(vector_store, all_chunks)
            if is_save:
                vector_store.save_local(folder_path=output_path, index_name=index_name)
                print("✅ [Success] FAISS vector DB 저장 성공.")

        elif db_type == "chroma":
            print(f"📌 [info] Vector DB type: {db_type}")
            chroma_path = os.path.join(output_path, index_name)
            if os.path.exists(chroma_path):
                import shutil
                shutil.rmtree(chroma_path)
                print(f"⚠️ [Notification] {db_type} 덮어쓰기 오류 방지를 위한 이전 생성 DB 제거...")

            vector_store = Chroma(
                embedding_function=embeddings,
                persist_directory=chroma_path,
                collection_name="chroma_db",
            )
            vector_store = add_docs_in_batch(vector_store, all_chunks)
            print("✅ [Success] Chroma vector DB 저장 성공.")

        else:
            raise ValueError("❌ [Value] (vector_db.generate_vector_db) 지원하지 않는 벡터 DB 타입입니다. ('faiss' 또는 'chroma' 사용)")

        return vector_store

    except Exception as e:
        raise RuntimeError(f"❌ [Runtime] (vector_db.generate_vector_db) 벡터 DB 생성 실패: {e}")


@trace(name="load_vector_db")
def load_vector_db(
    path: str,
    embed_model_name: str,
    index_name: str,
    db_type: str = "faiss"
) -> Union[FAISS, Chroma]:
    """
    저장된 FAISS 또는 Chroma 벡터 DB를 로컬 경로에서 로드합니다.

    Args:
        path (str): 저장된 벡터 DB의 루트 디렉토리 경로
        embed_model_name (str): 사용할 임베딩 모델 이름
        index_name (str): 불러올 벡터 DB의 인덱스 이름
        db_type (str): 벡터 DB 유형 ('faiss' 또는 'chroma')

    Returns:
        Union[FAISS, Chroma]: 로드된 벡터 DB 인스턴스

    Raises:
        FileNotFoundError: 지정한 경로가 존재하지 않을 경우
        ValueError: 지원하지 않는 DB 타입일 경우
        RuntimeError: 로딩 중 오류 발생 시
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ [FileNotFound] (vector_db.load_vector_db.path) 벡터 DB 경로가 존재하지 않습니다: {path}")

    try:
        embeddings = generate_embedding(embed_model_name)

        if db_type == "faiss":
            return FAISS.load_local(
                folder_path=path,
                index_name=index_name,
                embeddings=embeddings,
                allow_dangerous_deserialization=True,
            )
        elif db_type == "chroma":
            chroma_path = os.path.join(path, index_name)
            return Chroma(
                embedding_function=embeddings,
                persist_directory=chroma_path,
                collection_name="chroma_db",
            )
        else:
            raise ValueError(f"❌ [Value] (vector_db.load_vector_db) 지원하지 않는 벡터 DB 타입입니다. {db_type}")

    except Exception as e:
        raise RuntimeError(f"❌ [Runtime] (vector_db.load_vector_db) 벡터 DB 로드 실패: {e}")


def add_docs_in_batch(
    vector_store: VectorStore,
    chunks: Optional[List[Document]],
    batch_size: int = 128
):
    """
    주어진 문서 청크 리스트를 지정된 배치 크기(batch size)로 나누어 벡터 스토어에 추가합니다.

    Args:
        vector_store (VectorStore): 문서가 삽입될 벡터 저장소 객체
        chunks (Optional[List[Document]]): 삽입할 문서 청크 리스트
        batch_size (int): 한 번에 처리할 문서의 수

    Returns:
        VectorStore: 문서가 삽입된 벡터 저장소 인스턴스
    """
    total = len(chunks)
    pbar = tqdm(
        range(0, total, batch_size),
        desc=f"    📌 Indexing chunks to {vector_store.__class__.__name__}",
        unit="batch",
    )

    for i in pbar:
        batch = chunks[i:i + batch_size]
        vector_store.add_documents(batch)

        end_idx = min(i + batch_size, total)
        pbar.set_postfix_str(f"Indexed {end_idx} / {total}")

    return vector_store