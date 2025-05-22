from typing import List, Optional, Literal
from collections import defaultdict
from langchain.vectorstores.base import VectorStore
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from sklearn.metrics.pairwise import cosine_similarity
from src.embedding.vector_db import generate_embedding
from dotenv import load_dotenv
load_dotenv()

import re
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
from langchain.schema import Document

def compute_keyword_overlap_score(query: str, chunk: str) -> float:
    query_tokens = set(re.findall(r"\w+", query.lower()))
    chunk_tokens = set(re.findall(r"\w+", chunk.lower()))
    if not query_tokens or not chunk_tokens:
        return 0.0
    overlap = query_tokens.intersection(chunk_tokens)
    return len(overlap) / len(query_tokens)

def rerank_documents(
    query: str,
    docs: List[Document],
    embed_model,
    rerank_top_k: int,
    min_chunks: int,
    verbose: bool = False
) -> List[Document]:
    try:
        query_vec = [embed_model.embed_query(query)]
        doc_vecs = [embed_model.embed_documents([doc.page_content])[0] for doc in docs]
    except Exception as e:
        raise RuntimeError(f"❌ [Runtime] (retrieval.rerank_documents) 임베딩 실패: {e}")

    cosine_scores = cosine_similarity(query_vec, doc_vecs)[0]

    hybrid_scores = []
    for doc, cos_sim in zip(docs, cosine_scores):
        keyword_score = compute_keyword_overlap_score(query, doc.page_content)
        hybrid = 0.7 * cos_sim + 0.3 * keyword_score
        hybrid_scores.append((doc, hybrid))

    # 문서별로 그룹화
    doc_groups = defaultdict(list)
    for doc, hybrid_score in hybrid_scores:
        fname = doc.metadata.get("파일명")
        doc_groups[fname].append((doc, hybrid_score))

    if verbose:
        print("\n📌 [디버깅] 문서 그룹별 유사도 (cosine + keyword + hybrid):")
        for fname, group in doc_groups.items():
            print(f"\n📁 문서: {fname}")
            for doc, hybrid_score in group:
                idx = docs.index(doc)
                cos_sim = cosine_scores[idx]
                keyword_score = compute_keyword_overlap_score(query, doc.page_content)
                print(f"  - chunk_idx: {doc.metadata.get('chunk_idx')}, "
                    f"cosine: {cos_sim:.4f}, keyword: {keyword_score:.4f}, hybrid: {hybrid_score:.4f}")

    # 문서별 평균 hybrid 유사도 계산
    doc_scores = []
    for fname, group in doc_groups.items():
        avg_score = sum(score for _, score in group) / len(group)
        doc_scores.append((fname, avg_score))

    if verbose:
        print("\n📌 [디버깅] 문서별 평균 hybrid 유사도:")
        for rank, (fname, score) in enumerate(sorted(doc_scores, key=lambda x: x[1], reverse=True), 1):
            print(f"  {rank}. {fname} (평균 hybrid 유사도: {score:.4f})")

    # 상위 문서 선택
    top_doc_fnames = set(fname for fname, _ in sorted(doc_scores, key=lambda x: x[1], reverse=True)[:rerank_top_k])

    if verbose:
        print(f"\n✅ 선택된 상위 {rerank_top_k}개 문서:")
        for fname in top_doc_fnames:
            print(f"  - {fname}")

    # 각 문서에서 min_chunks 만큼 청크 선택
    selected_docs = []
    for fname in top_doc_fnames:
        group = sorted(doc_groups[fname], key=lambda x: x[1], reverse=True)
        selected = group[:min_chunks]
        selected_docs.extend([doc for doc, _ in selected])

        if verbose:
            print(f"\n📄 문서 {fname}의 선택된 상위 {min_chunks}개 청크:")
            for doc, score in selected:
                print(f"  - chunk_idx: {doc.metadata.get('chunk_idx')}, hybrid 유사도: {score:.4f}")

    return selected_docs


def retrieve_documents(
    query: str,
    vector_store: VectorStore,
    top_k: int,
    search_type: Literal["similarity", "hybrid"],
    chunks: Optional[List[Document]],
    embed_model_name: str,
    rerank: bool,
    rerank_top_k: int,
    min_chunks: int,
    verbose: bool = False
) -> List[Document]:
    """
    TODO:
    - 리트리버는 벡터DB에서 검색해 청크를 가져오는데, 문서마다 리트리버가 적용되게 만들어 최소 청크 수를 보장해야 할 것 같다.

    주어진 쿼리에 대해 similarity 또는 hybrid 검색 방식으로 관련 문서를 검색합니다.
    리랭크가 비활성화되어도 문서마다 최소 청크 수를 확보하는 후처리를 포함합니다.

    Args:
        query (str): 사용자 검색 쿼리
        vector_store (VectorStore): 벡터 저장소 인스턴스
        top_k (int): 검색할 최대 문서 개수
        search_type (Literal["similarity", "hybrid"]): 검색 방식
        chunks (Optional[List[Document]]): hybrid 검색을 위한 전체 문서 리스트
        embed_model_name (str): 사용할 임베딩 모델 이름
        rerank (bool): re-ranking 적용 여부
        min_chunks (int): 문서마다 보장되는 최소 청크 수
        verbose (bool): 디버그 출력 여부

    Returns:
        List[Document]: 검색 또는 재정렬된 문서 리스트

    Raises:
        RuntimeError: retriever 생성에 실패할 경우
        ValueError: 지원하지 않는 검색 방식 또는 chunks 미제공 시
    """
    
    try:
        embed_model = generate_embedding(embed_model_name=embed_model_name)
    except Exception as e:
        raise RuntimeError(f"❌ [Runtime] (retrieval.retrieve_documents.embed_model) 임베딩 모델 생성 실패: {e}")
        
    if search_type == "similarity":
        try:
            docs = vector_store.similarity_search(query, k=top_k)
        except Exception as e:
            raise RuntimeError(f"❌ similarity_search 실패: {e}")
        
    elif search_type == "hybrid":
        if chunks is None:
            raise ValueError("❌ [Value] (retrieval.retrieve_documents.chunks) hybrid 검색을 위해 chunks가 필요합니다.")
        try:
            vector_retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": top_k}
            )
        except Exception as e:
            raise RuntimeError(f"❌ [Runtime] (retrieval.retrieve_documents.vector_retriever) FAISS retriever 생성 실패: {e}")
    
        try:
            bm25_retriever = BM25Retriever.from_documents(chunks)
            bm25_retriever.k = top_k
        except Exception as e:
            raise RuntimeError(f"❌ [Runtime] (retrieval.retrieve_documents.bm25_retriever) BM25 retriever 생성 실패: {e}")
        
        hybrid_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.5, 0.5]
        )
        docs = hybrid_retriever.invoke(query)
        
        seen_pairs = set()
        unique_docs = []
        for doc in docs:
            identifier = (doc.metadata.get("파일명"), doc.page_content.strip())
            if identifier not in seen_pairs:
                unique_docs.append(doc)
                seen_pairs.add(identifier)
        docs = unique_docs[:top_k]
        
    else:
        raise ValueError(f"❌ [Value] (retrieval.retrieve_documents.search_type) 지원하지 않는 검색 방식입니다: {search_type}")

    if rerank:
        docs = rerank_documents(query, docs, embed_model, rerank_top_k, min_chunks, verbose)

    return docs