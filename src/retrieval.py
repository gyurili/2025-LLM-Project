from typing import List, Dict, Optional
from langchain.vectorstores.base import VectorStore
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever


def retrieve_documents(
    query: str,
    vector_store: VectorStore,
    top_k: int,
    search_type: str,
    all_documents: Optional[List[Document]],
    embeddings
) -> List[Dict]:

    if search_type == "similarity":
        docs = vector_store.similarity_search(query, k=top_k)

    elif search_type == "hybrid":
        # FAISS
        vector_retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k}
        )
        # BM25
        bm25_retriever = BM25Retriever.from_documents(all_documents)
        bm25_retriever.k = top_k
        
        hybrid_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.5, 0.5]
        )
        docs = hybrid_retriever.get_relevant_documents(query)
    else:
        raise ValueError(f"❌ 지원하지 않는 검색 방식입니다: {search_type}")

    return [{"text": doc.page_content, "metadata": doc.metadata} for doc in docs]
