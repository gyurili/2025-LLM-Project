import os
from typing import List, Union, Optional, Literal

import faiss
from dotenv import load_dotenv
from tqdm import tqdm

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
    임베딩 모델을 초기화합니다.

    Args:
        embed_model_name (str): 'openai' 또는 huggingface 모델 이름

    Returns:
        임베딩 객체

    Raises:
        ValueError: 모델 초기화 실패 시
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


def generate_vector_db(
    all_chunks: List[Document],
    embed_model_name: str,
    index_name: str,
    db_type: str = "faiss",
    is_save: bool = True,
    batch_size: int = 128
) -> Union[FAISS, Chroma]:
    """
    FAISS 또는 Chroma 기반 벡터 DB를 생성하고 로컬에 저장합니다.
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


def load_vector_db(
    path: str,
    embed_model_name: str,
    index_name: str,
    db_type: str = "faiss"
) -> Union[FAISS, Chroma]:
    """
    로컬에서 저장된 FAISS 또는 Chroma 벡터 DB를 로드합니다.
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
    문서 chunk를 batch별로 추가하는 방식.
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