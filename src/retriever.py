from typing import List, Dict
from langchain.vectorstores.base import VectorStore


def search_documents(
    question: str,
    vector_store: VectorStore,
    top_k: int = 3,
    search_type: str = "similarity"
) -> List[Dict]:
    """
    외부에서 주어진 벡터 스토어를 사용해 질문에 유사한 문서를 검색합니다.

    Args:
        question (str): 검색할 질문 문장
        vector_store (VectorStore): 검색에 사용할 벡터 DB 객체 (예: FAISS)
        top_k (int): 검색 결과 개수
        search_type (str): 검색 방식 ('similarity', 'mmr' 등)

    Returns:
        List[Dict]: 검색된 문서들의 리스트 (본문 및 메타데이터 포함)
    """
    # 검색기(retriever) 구성
    retriever = vector_store.as_retriever(
        search_type=search_type,
        search_kwargs={"k": top_k}
    )

    # 질문에 유사한 문서 검색
    docs = retriever.invoke(question)

    # 결과 형식 정리
    results = []
    for doc in docs[:top_k]:
        results.append({
            "text": doc.page_content.strip(),
            "source": doc.metadata.get("파일명", "unknown"),
            "기관": doc.metadata.get("발주 기관", "unknown"),
            "사업명": doc.metadata.get("사업명", "unknown"),
            "chunk_idx": doc.metadata.get("chunk_idx", -1)
        })

    return results