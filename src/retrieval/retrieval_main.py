from src.retrieval.retrieval import retrieve_documents
from src.embedding.vector_db import load_vector_db
from src.embedding.embedding_main import generate_index_name


def retrieval_main(config, vector_store, chunks):
    """
    설정에 따라 similarity 또는 hybrid 방식으로 검색하고, 필요시 re-ranking을 수행합니다.

    Args:
        config (dict): 설정 정보
        vector_store (VectorStore or None): 기존 로드된 벡터 저장소, 없으면 로드
        chunks (List[Document]): hybrid 검색 시 사용할 전체 문서 청크

    Returns:
        List[Document]: 검색된 문서의 내용과 메타데이터를 담은 Document 리스트
    """
    index_name = generate_index_name(config)

    vector_db_path = config.get("embedding", {}).get("vector_db_path", "data")
    embed_model = config.get("embedding", {}).get("embed_model", "openai")
    db_type = config.get("embedding", {}).get("db_type", "faiss")

    if vector_store is None:
        vector_store=load_vector_db(vector_db_path, embed_model, index_name, db_type)
        if config.get("settings", {}).get("verbose", False):
            print("✅ Vector DB 로드 완료")
            
    query = config.get("retriever", {}).get("query", "")
    top_k = config.get("retriever", {}).get("top_k", 5)
    search_type = config.get("retriever", {}).get("search_type", "similarity")
    rerank = config.get("retriever", {}).get("rerank", True)
    rerank_top_k = config.get("retriever", {}).get("rerank_top_k", 3)
    verbose = config.get("settings", {}).get("verbose", False)

    docs = retrieve_documents(query, vector_store, top_k, search_type, chunks, embed_model, rerank, rerank_top_k)
    if verbose:
        print(f"\n- 임베딩 모델: {embed_model}")
        print(f"- DB 타입: {db_type}")
        print(f"- 벡터 DB 파일: {index_name}")
        print(f"- 청킹 검색 type: {search_type}")
        print(f"- reRank 적용 여부: {rerank}")
        for i, doc in enumerate(docs, 1):
            print(f"\n📄 문서 {i}")
            print(f"본문:\n{doc.page_content[:300]}...")
            print(f"메타데이터: {doc.metadata}")
            
    return docs