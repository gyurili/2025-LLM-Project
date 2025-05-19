from typing import List, Dict, Optional, Literal
from langchain.vectorstores.base import VectorStore
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

def rerank_documents(
    query: str,
    docs: List[Document],
    embed_model,
    rerank_top_k: int
    ) -> List[Document]:
    
    query_vec = embed_model.embed_query(query)
    doc_vecs = embed_model.embed_documents([doc.page_content for doc in docs])
    
    query_vec = [query_vec]
    similarities = cosine_similarity(query_vec, doc_vecs)[0]
    
    doc_scores = [(doc, score) for doc, score in zip(docs, similarities)]
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    
    return [doc for doc, _ in doc_scores[:rerank_top_k]]


def retrieve_documents(
    query: str,
    vector_store: VectorStore,
    top_k: int,
    search_type: Literal["similarity", "hybrid"],
    chunks: Optional[List[Document]],
    embed_model_name: str,
    rerank: bool,
    rerank_top_k: int,
) -> List[Document]:
    """
    주어진 쿼리에 대해 지정된 검색 방식으로 관련 문서를 검색합니다.

    Args:
        query (str): 사용자 검색 쿼리
        vector_store (VectorStore): 벡터 저장소 인스턴스
        top_k (int): 검색할 최대 문서 개수
        search_type (str): 검색 방식 ("similarity" 또는 "hybrid")
        chunks (Optional[List[Document]]): hybrid 검색을 위한 전체 문서

    Returns:
        List[Document]: 검색된 문서의 내용과 메타데이터를 담은 Document 리스트

    Raises:
        RuntimeError: retriever 생성에 실패할 경우
        ValueError: 지원하지 않는 검색 방식일 경우
    """
    embed_model = HuggingFaceEmbeddings(model_name=embed_model_name)
    
    if search_type == "similarity":
        docs = vector_store.similarity_search(query, k=top_k)
        if rerank:
            docs = rerank_documents(query, docs, embed_model, rerank_top_k)
        
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
        
        if rerank:
            docs = rerank_documents(query, docs, embed_model, rerank_top_k)
        else:
            docs = docs[:top_k]
    else:
        raise ValueError(f"❌ [Value] (retrieval.retrieve_documents.search_type) 지원하지 않는 검색 방식입니다: {search_type}")

    return docs