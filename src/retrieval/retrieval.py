from typing import List, Optional
from langsmith import traceable
from langchain.schema import Document
from langchain.vectorstores.base import VectorStore
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder

from src.generator.generator_main import load_chat_history

@traceable(name="rerank_documents")
def rerank_documents(
    query: str,
    docs: List[Document],
    rerank_top_k: int,
    verbose: bool
) -> List[Document]:
    """
    검색어와 문서 간 CrossEncoder 점수를 기반으로 문서를 재정렬하여 상위 N개를 반환합니다.

    Args:
        query (str): 사용자 검색 쿼리
        docs (List[Document]): 검색으로 추출된 문서 리스트
        rerank_top_k (int): 최종 반환할 문서 개수
        verbose (bool): 로그 출력 여부

    Returns:
        List[Document]: 재정렬된 상위 문서 리스트
    """
    if verbose:
        print("\n📌 기존 문서 순서:")
        for i, doc in enumerate(docs, 1):
            print(f"  {i}. 파일명: {doc.metadata.get('파일명')}, 청크: {doc.metadata.get('chunk_idx')}")

    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = [(query, doc.page_content) for doc in docs]
    scores = model.predict(pairs)

    doc_scores = [(doc, score) for doc, score in zip(docs, scores)]
    doc_scores.sort(key=lambda x: x[1], reverse=True)

    if verbose:
        print("\n📌 re-rank 적용 후 문서 순서:")
        for i, (doc, score) in enumerate(doc_scores, 1):
            print(f"  {i}. 파일명: {doc.metadata.get('파일명')}, 청크: {doc.metadata.get('chunk_idx')}, 점수: {score:.4f}")
    else:
        print("\n📌 최종 문서 순서(상위 5개):")
        for i, (doc, score) in enumerate(doc_scores[:5], 1):
            print(f"  {i}. 파일명: {doc.metadata.get('파일명')}, 청크: {doc.metadata.get('chunk_idx')}, 점수: {score:.4f}")

    return [doc for doc, _ in doc_scores[:rerank_top_k]]


@traceable(name="retrieve_documents")
def retrieve_documents(
    vector_store: VectorStore,
    chunks: Optional[List[Document]],
    config: dict
) -> List[Document]:
    """
    주어진 쿼리에 대해 similarity 또는 hybrid 검색 방식으로 관련 문서를 검색합니다.

    Args:
        vector_store (VectorStore): 벡터 저장소 인스턴스
        chunks (Optional[List[Document]]): hybrid 검색을 위한 전체 문서 리스트
        config (dict): 설정 config

    Returns:
        List[Document]: 검색 또는 재정렬된 문서 리스트

    Raises:
        RuntimeError: retriever 생성에 실패할 경우
        ValueError: 지원하지 않는 검색 방식 또는 chunks 미제공 시
    """
    query = config['retriever']['query']
    search_type = config['retriever']['search_type']
    top_k = config['retriever']['top_k']
    rerank = config['retriever']['rerank']
    rerank_top_k = config['retriever']['rerank_top_k']
    verbose = config['settings']['verbose']

    # 과거 질의응답 내역 불러오기
    chat_history = load_chat_history(config)
    query = f"맥락: {chat_history}\n 질문:{query}"
    
    if search_type == "similarity":
        docs = vector_store.similarity_search(query, k=top_k)
    elif search_type == "hybrid":
        if chunks is None:
            raise ValueError("❌ [Value] (retrieval.retrieve_documents.chunks) chunks 누락 오류.")

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
            weights=[0.4, 0.6]
        )
        docs = hybrid_retriever.invoke(query)

        seen_pairs = set()
        unique_docs = []
        for doc in docs:
            identifier = (doc.metadata.get("파일명"), doc.metadata.get("chunk_idx"))
            if identifier not in seen_pairs:
                unique_docs.append(doc)
                seen_pairs.add(identifier)
        docs = unique_docs[:top_k]
    else:
        raise ValueError(f"❌ [Value] (retrieval.retrieve_documents.search_type) search_type 값 오류: {search_type}")
    
    if rerank:
        docs = rerank_documents(query, docs, rerank_top_k, verbose)
    else:
        docs = docs[:top_k]

    return docs