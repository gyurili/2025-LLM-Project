import os
from typing import List, Union

import faiss
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

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
        raise ValueError(f"임베딩 모델 초기화 실패: {e}")

def generate_vector_db(all_chunks: List[Document], embed_model_name: str) -> Union[OpenAIEmbeddings, HuggingFaceEmbeddings]:
    """
    FAISS 기반 벡터 DB를 생성하고 로컬에 저장합니다.

    Args:
        all_chunks (List[Document]): 청크된 문서 리스트
        embed_model_name (str): 사용할 임베딩 모델 이름

    Returns:
        임베딩 객체

    Raises:
        ValueError: 임베딩 처리 실패 시
    """
    embeddings = generate_embedding(embed_model_name)
    try:
        dimension = len(embeddings.embed_query("hello world"))
    except Exception as e:
        raise ValueError(f"임베딩 차원 계산 실패: {e}")

    vector_store = FAISS(
        embedding_function=embeddings,
        index=faiss.IndexFlatL2(dimension),
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )
    vector_store.add_documents(all_chunks)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(base_dir, "data")
    
    vector_store.save_local(folder_path=output_path, index_name="hwp_faiss_index")
    return embeddings

def load_vector_db(path: str, embed_model_name: str) -> FAISS:
    """
    로컬에서 저장된 FAISS 벡터 DB를 로드합니다.

    Args:
        path (str): 벡터 DB가 저장된 루트 경로 (예: "data")
        embed_model_name (str): 임베딩 모델 이름

    Returns:
        FAISS: 로드된 벡터 DB 객체

    Raises:
        FileNotFoundError: 경로가 없을 경우
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"벡터 DB 경로가 존재하지 않습니다: {path}")

    embeddings = generate_embedding(embed_model_name)

    return FAISS.load_local(
        folder_path=path,
        index_name="hwp_faiss_index",
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )