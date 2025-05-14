from langchain.vectorstores import FAISS
from langchain.schema import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List


def simple_similarity_search(vector_store: FAISS, query: str, k: int = 5) -> List[Document]:
    return vector_store.similarity_search(query, k=k)


def hybrid_search(vector_store: FAISS, query: str, k: int = 5) -> List[Document]:
    """
    벡터 기반 + 키워드 기반 TF-IDF 검색을 혼합한 Hybrid 검색

    Args:
        vector_store (FAISS): 로드된 벡터 DB
        query (str): 사용자의 검색 쿼리
        k (int): 반환할 문서 수

    Returns:
        List[Document]: 혼합된 유사 문서 리스트
    """
    # 벡터 기반 검색
    vector_results = vector_store.similarity_search(query, k=k)

    # 전체 문서 가져오기 (TF-IDF용)
    all_docs = list(vector_store.docstore._dict.values())
    corpus = [doc.page_content for doc in all_docs]

    # TF-IDF 기반 검색
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    query_vec = vectorizer.transform([query])
    tfidf_scores = (tfidf_matrix @ query_vec.T).toarray().ravel()

    # 점수 높은 k개 문서 인덱스
    top_indices = tfidf_scores.argsort()[::-1][:k]
    tfidf_results = [all_docs[i] for i in top_indices]

    # 벡터 + TF-IDF 결과 합치기 (중복 제거)
    combined_docs = {doc.page_content: doc for doc in (vector_results + tfidf_results)}

    # 최대 k개 반환
    return list(combined_docs.values())[:k]


def retrieve(query: str, vector_store: FAISS, k: int, method: str = "simple") -> List[Document]:
    """
    검색 방식에 따라 검색 실행
    """
    if method == "simple":
        return simple_similarity_search(vector_store, query, k)
    elif method == "hybrid":
        return hybrid_search(vector_store, query, k)
    else:
        raise ValueError(f"지원하지 않는 검색 방식입니다: {method}")
