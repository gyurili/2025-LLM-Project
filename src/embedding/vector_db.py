import os
from typing import List, Union

import faiss
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

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
            return OpenAIEmbeddings()
        else:
            return HuggingFaceEmbeddings(model_name=embed_model_name)
    except Exception as e:
        raise ValueError(f"❌ [Value] (vector_db.generate_embedding) 임베딩 모델 초기화 실패: {e}")

def generate_vector_db(
        all_chunks: List[Document], 
        embed_model_name: str,
        index_name: str,
        db_type: str = "faiss"
    ) -> Union[FAISS, Chroma]:
    """
    FAISS 기반 벡터 DB를 생성하고 로컬에 저장합니다.

    Args:
        all_chunks (List[Document]): 청크된 문서 리스트
        embed_model_name (str): 사용할 임베딩 모델 이름
        index_name (str): 벡터 DB 인덱스 이름
        db_type (str): 사용할 벡터 DB 타입 ('faiss' 또는 'chroma')

    Returns:
        VectorStore 객체

    Raises:
        ValueError: 임베딩 처리 실패 시
    """
    if not all_chunks or not isinstance(all_chunks, list):
        raise ValueError("❌ [Value] (vector_db.generate_vector_db) all_chunks는 비어 있지 않은 Document 리스트여야 합니다.")

    embeddings = generate_embedding(embed_model_name)
    try:
        dimension = len(embeddings.embed_query("hello world"))
    except Exception as e:
        raise ValueError(f"❌ [Value] (vector_db.generate_vector_db) 임베딩 차원 계산을 실패했습니다.: {e}")

    try:
        output_path = os.path.join(get_project_root_dir(), "data")
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
        
        if db_type == "faiss":
            vector_store = FAISS(
                embedding_function=embeddings,
                index=faiss.IndexFlatL2(dimension),
                docstore=InMemoryDocstore(),
                index_to_docstore_id={}
            )
            vector_store.add_documents(all_chunks)
            vector_store.save_local(folder_path=output_path, index_name=index_name)
        elif db_type == "chroma":
            chroma_path = os.path.join(output_path, index_name)
            vector_store = Chroma.from_documents(
                documents=all_chunks,
                embedding=embeddings,
                persist_directory=chroma_path,
            )
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
    로컬에서 저장된 FAISS 벡터 DB를 로드합니다.

    Args:
        path (str): 벡터 DB가 저장된 루트 경로 (예: "data")
        embed_model_name (str): 임베딩 모델 이름
        index_name (str): 벡터 DB 인덱스 이름
        db_type (str): 벡터 DB 타입 ('faiss' 또는 'chroma')

    Returns:
        VectorStore 객체

    Raises:
        FileNotFoundError: 경로가 없을 경우
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
            )
        else:
            raise ValueError(f"❌ [Value] (vector_db.load_vector_db) 지원하지 않는 벡터 DB 타입입니다. {db_type}")
        
    except Exception as e:
        raise RuntimeError(f"❌ [Runtime] (vector_db.load_vector_db) 벡터 DB 로드 실패: {e}")