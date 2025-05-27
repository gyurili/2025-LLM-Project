from typing import List
from langsmith import trace, traceable
from langchain.schema import Document

from src.loader.data_loader import retrieve_top_documents_from_metadata, data_process
from src.loader.splitter import data_chunking, summarize_chunk_quality


@traceable(name="loader_main")
def loader_main(config, embeddings, chat_history) -> List[Document]:
    """
    설정 정보를 기반으로 문서를 로드하고, 전처리 및 청크 작업을 수행합니다.

    Args:
        config (dict): 시스템 설정 정보를 담은 딕셔너리
        embedder: 사전 생성된 임베딩 모델 인스턴스

    Returns:
        List[Document]: 처리된 문서의 청크 리스트
    """
    data_config = config["data"]
    query = config["retriever"]["query"]
    top_k = config["data"]["top_k"]
    verbose = config["settings"]["verbose"]

    # 데이터 로드
    with trace(name="load_data"):
        data_list_path = data_config.get("data_list_path", "data/data_list.csv")
        df = retrieve_top_documents_from_metadata(
            query=query,
            csv_path=data_list_path,
            embeddings=embeddings,
            chat_history=chat_history,
            top_k=top_k,
        )
        print("✅ 문서 유사도 검색 완료")

    # 데이터 전처리
    with trace(name="process_data"):
        file_type = config["data"]["file_type"]
        apply_ocr = config["data"]["apply_ocr"]
        df = data_process(df, config=config, apply_ocr=apply_ocr, file_type=file_type)
        print("✅ 데이터 전처리 완료")

    # 청크 생성
    with trace(name="chunk_documents"):
        splitter_type = config["data"]["splitter"]
        chunk_size = config["data"]["chunk_size"]
        chunk_overlap = config["data"]["chunk_overlap"]

        chunks = data_chunking(
            df=df,
            splitter_type=splitter_type,
            size=chunk_size,
            overlap=chunk_overlap,
        )
        print("✅ 청크 생성 완료")

    # 청크 품질 검사
    summarize_chunk_quality(chunks, verbose)
    if verbose:
        print("✅ 청크 품질 검사 완료")

    return chunks