from typing import List

from langchain.schema import Document

from src.loader.data_loader import (
    retrieve_top_documents_from_metadata,
    data_process,
)
from src.loader.splitter import (
    data_chunking,
    summarize_chunk_quality,
)


def loader_main(config: dict) -> List[Document]:
    """
    설정 정보를 기반으로 문서를 로드하고, 전처리 및 청크 작업을 수행합니다.

    Args:
        config (dict): 시스템 설정 정보를 담은 딕셔너리

    Returns:
        List[Document]: 처리된 문서의 청크 리스트
    """
    data_config = config.get("data", {})
    query = config.get("retriever", {}).get("query", "사업")
    top_k = data_config.get("top_k", 5)
    verbose = config.get("settings", {}).get("verbose", True)
    embed_model = config.get("embedding", {}).get("embed_model", "openai")

    # 1. 데이터 로드
    data_list_path = data_config.get("data_list_path", "data/data_list.csv")
    df = retrieve_top_documents_from_metadata(
        query=query,
        csv_path=data_list_path,
        embed_model=embed_model,
        top_k=top_k,
    )
    print("✅ 문서 유사도 검색 완료")

    # 2. 데이터 전처리
    file_type = data_config.get("file_type", "all")
    apply_ocr = data_config.get("apply_ocr", False)
    df = data_process(df, apply_ocr=apply_ocr, file_type=file_type)
    print("✅ 데이터 전처리 완료")

    # 3. 청크 생성
    splitter_type = data_config.get("splitter", "section")
    chunk_size = data_config.get("chunk_size", 1000)
    chunk_overlap = data_config.get("chunk_overlap", 250)

    chunks = data_chunking(
        df=df,
        splitter_type=splitter_type,
        size=chunk_size,
        overlap=chunk_overlap,
    )
    print("✅ 청크 생성 완료")

    # 4. 청크 품질 검사
    summarize_chunk_quality(chunks, verbose)
    if verbose:
        print("✅ 청크 품질 검사 완료")

    return chunks