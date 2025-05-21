from typing import List, Optional, Literal
from collections import defaultdict, OrderedDict
from langchain.vectorstores.base import VectorStore
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from src.embedding.vector_db import generate_embedding

import os
from dotenv import load_dotenv
load_dotenv()

def rerank_documents(
    query: str,
    docs: List[Document],
    embed_model,
    min_chunks: int,
    max_chunks: Optional[int] = None,
    verbose: bool = False
    ) -> List[Document]:
    """
        TODO:
        - 검색된 문서마다 최소 청크 수를 보장하는 로직을 추가합니다.

    검색어와 문서 간 임베딩 유사도를 기반으로 문서를 재정렬하여 상위 N개를 반환합니다.

    Args:
        query (str): 사용자 검색 쿼리
        docs (List[Document]): 검색으로 추출된 문서 리스트
        embed_model: 임베딩 모델 객체
        min_chunks (int): 문서별 보장되는 최소 청크 수

    Returns:
        List[Document]: 유사도 기준으로 재정렬된 상위 문서 리스트
    """
    if verbose:
        print("\n📌 기존 문서 순서:")
        for i, doc in enumerate(docs, 1):
            print(f"  {i}. 파일명: {doc.metadata.get('파일명')}, 청크: {doc.metadata.get('chunk_idx')}")
    
    query_vec = embed_model.embed_query(query)
    doc_vecs = embed_model.embed_documents([doc.page_content for doc in docs])
    
    query_vec = [query_vec]
    similarities = cosine_similarity(query_vec, doc_vecs)[0]
    
    doc_scores = [(doc, score) for doc, score in zip(docs, similarities)]
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    
    if verbose:
        print("\n📌 re-rank 적용 후 문서 순서:")
        for i, (doc, score) in enumerate(doc_scores, 1):
            print(f"  {i}. 파일명: {doc.metadata.get('파일명')}, 청크: {doc.metadata.get('chunk_idx')}, 유사도: {score:.4f}")
    
    grouped = OrderedDict()
    for doc, score in doc_scores:
        fname = doc.metadata.get("파일명")
        if fname not in grouped:
            grouped[fname] = []
        grouped[fname].append((doc, score))

    selected_set = set()
    selected_docs = []

    for fname, group in grouped.items():
        limit = max_chunks if max_chunks else len(group)
        group_sorted = sorted(group, key=lambda x: x[1], reverse=True)

        count = 0
        for doc, _ in group_sorted:
            doc_id = (doc.metadata.get("파일명"), doc.metadata.get("chunk_idx"))
            if doc_id not in selected_set:
                selected_docs.append(doc)
                selected_set.add(doc_id)
                count += 1
            if count >= max(min_chunks, limit):
                break

    if verbose:
        print("\n📌 최종 선택된 문서:")
        for i, doc in enumerate(selected_docs, 1):
            print(f"  {i}. 파일명: {doc.metadata.get('파일명')}, 청크: {doc.metadata.get('chunk_idx')}")

    return selected_docs


def retrieve_documents(
    query: str,
    vector_store: VectorStore,
    top_k: int,
    search_type: Literal["similarity", "hybrid"],
    chunks: Optional[List[Document]],
    embed_model_name: str,
    rerank: bool,
    min_chunks: int,
    verbose: bool = False
) -> List[Document]:
    """
        TODO:
        - 문서 그루핑 및 대표 점수 계산을 통해 유사한 문서 그룹을 선택하는 로직을 개선합니다.

    주어진 쿼리에 대해 similarity 또는 hybrid 검색 방식으로 관련 문서를 검색합니다.

    Args:
        query (str): 사용자 검색 쿼리
        vector_store (VectorStore): 벡터 저장소 인스턴스
        top_k (int): 검색할 최대 문서 개수
        search_type (Literal["similarity", "hybrid"]): 검색 방식
        chunks (Optional[List[Document]]): hybrid 검색을 위한 전체 문서 리스트
        embed_model_name (str): 사용할 임베딩 모델 이름
        rerank (bool): re-ranking 적용 여부
        min_chunks (int): 문서마다 보장되는 최소 청크 수

    Returns:
        List[Document]: 검색 또는 재정렬된 문서 리스트

    Raises:
        RuntimeError: retriever 생성에 실패할 경우
        ValueError: 지원하지 않는 검색 방식 또는 chunks 미제공 시
    """
    
    embed_model = generate_embedding(embed_model_name=embed_model_name)
    
    if search_type == "similarity":
        docs = vector_store.similarity_search(query, k=top_k * 5)
        
    elif search_type == "hybrid":
        if chunks is None:
            raise ValueError("❌ [Value] (retrieval.retrieve_documents.chunks) hybrid 검색을 위해 chunks가 필요합니다.")
        try:
            vector_retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": top_k * 5}
            )
        except Exception as e:
            raise RuntimeError(f"❌ [Runtime] (retrieval.retrieve_documents.vector_retriever) FAISS retriever 생성 실패: {e}")
    
        try:
            bm25_retriever = BM25Retriever.from_documents(chunks)
            bm25_retriever.k = top_k * 5
        except Exception as e:
            raise RuntimeError(f"❌ [Runtime] (retrieval.retrieve_documents.bm25_retriever) BM25 retriever 생성 실패: {e}")
        
        hybrid_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.5, 0.5]
        )
        docs = hybrid_retriever.invoke(query)
        
    else:
        raise ValueError(f"❌ [Value] (retrieval.retrieve_documents.search_type) 지원하지 않는 검색 방식입니다: {search_type}")
    
    # 문서 그룹화 및 각 문서당 최소 청크 수 보장
    doc_groups = defaultdict(list)
    for doc in docs:
        fname = doc.metadata.get("파일명")
        doc_groups[fname].append(doc)

    query_vec = embed_model.embed_query(query)
    selected_docs = []
    for fname, group in doc_groups.items():
        if len(group) > 1:
            doc_vecs = embed_model.embed_documents([doc.page_content for doc in group])
            similarities = cosine_similarity([query_vec], doc_vecs)[0]
            ranked_group = sorted(zip(group, similarities), key=lambda x: x[1], reverse=True)
            selected_docs.extend([doc for doc, _ in ranked_group])
        else:
            selected_docs.append(group[0])

    docs = selected_docs

    if rerank:
        docs = rerank_documents(query, docs, embed_model, min_chunks, max_chunks=5, verbose=verbose)

    return docs