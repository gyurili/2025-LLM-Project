from langchain.vectorstores.base import VectorStore
from typing import List, Dict


def retrieve_documents(
    question: str,
    vector_store: VectorStore,
    top_k: int = 3,
    search_type: str = "similarity"
) -> List[Dict]:

    if search_type == "similarity":
        docs = vector_store.similarity_search(question, k=top_k)
    elif search_type == "hybrid":
        # 구현 필요
        docs = vector_store.similarity_search(question, k=top_k)
    else:
        raise ValueError(f"❌ 지원하지 않는 검색 방식입니다: {search_type}")

    return [{"text": doc.page_content, "metadata": doc.metadata} for doc in docs]
