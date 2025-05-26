from typing import List, Optional

from langsmith import traceable
from langchain.schema import Document

from src.retrieval.retrieval import retrieve_documents
from src.embedding.vector_db import load_vector_db
from src.embedding.embedding_main import generate_index_name

@traceable(name="retrieval_main")
def retrieval_main(
    config: dict,
    vector_store: Optional[object],
    chunks: List[Document]
) -> List[Document]:
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

    vector_db_path = config["embedding"]["vector_db_path"]
    embed_model = ["embedding"]["embed_model"]
    db_type = config["embedding"]["db_type"]

    if vector_store is None:
        ### 인자 수정 필요
        vector_store = load_vector_db(
            path=vector_db_path,
            embed_model_name=embed_model,
            index_name=index_name,
            db_type=db_type
        )

    docs = retrieve_documents(
        vector_store=vector_store,
        chunks=chunks,
        config=config
    )
    
    verbose = config["settings"]["verbose"]

    if verbose:
        count = 0
        for i, doc in enumerate(docs, 1):
            print(f"\n📄 문서 {i}")
            print(f"본문:\n{doc.page_content[:100]}...")
            print(f"메타데이터: {doc.metadata}")
            count += 1
            if count > 4:
                break

    return docs