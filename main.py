import os
import yaml

from src.data_load import data_load, data_process, data_chunking
from src.vector_db import generate_vector_db, load_vector_db


if __name__ == "__main__":
    # Config 불러오기
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config["settings"].get("verbose", False):
        print("Verbose 모드로 실행 중입니다.")
        print(config)

    # 경로 준비
    folder_path = os.path.abspath(config["data"]["folder_path"])
    data_list_path = os.path.abspath(config["data"]["data_list_path"])
    file_type = config["data"]["type"]
    apply_ocr = config["data"].get("apply_ocr", False)

    # 데이터 로딩 및 전처리
    df = data_load(data_list_path)
    df = data_process(df, apply_ocr=apply_ocr, file_type=file_type)

    # 청크 분할
    all_chunks = data_chunking(df)
    print("청크 분할 완료!")

    # 벡터 DB 생성 및 저장
    embed_model_name = config["embedding"]["model"]
    vector_db_path = os.path.abspath(config["embedding"]["vector_db_path"])

    generate_vector_db(all_chunks, embed_model_name)
    print("벡터 DB 저장 완료!")

    # 벡터 DB 로드
    vector_store = load_vector_db(vector_db_path, embed_model_name)
    print("벡터 DB 로드 완료!")

    # 유사도 검색
    query_text = config["query"]["query"]
    top_k = config["query"]["top_k"]

    print(f"\n질문: {query_text}")
    print(f"유사도 검색 결과 (상위 {top_k}개 문서):")

    docs = vector_store.similarity_search(query_text, k=top_k)
    for i, doc in enumerate(docs, start=1):
        print(f"\n문서 {i}:\n{doc.page_content}")



# --------------------------------------------
# 영선님 작업물
'''
    if config["data"]["type"] == "hwp":
        df = data_process(df)
        all_chunks = hwp_chunking(df)
    elif config["data"]["type"] == "pdf":
        df = process_all_pdfs_in_folder(folder_path, apply_ocr=config["data"]["apply_ocr"])
    else:
        raise ValueError("지원하지 않는 데이터 타입입니다. 'hwp' 또는 'pdf'를 선택하세요.")

    # 벡터 DB 생성
    embeddings = generate_vector_db(all_chunks, config["embedding"]["model"])
    print("Vector DB가 저장되었습니다!")

    # 벡터 DB 로드
    vector_store = load_vector_db(config["embedding"]["vector_db_path"], embeddings)
    print("Vector DB가 로드되었습니다!")

    # 유사도 검색
    query = config["query"]["text"]
    k = config["query"]["top_k"]
    print(f"질문: {query}")
    print(f"유사도 검색 결과 (상위 {k}개 문서):")

    docs = vector_store.similarity_search(query, k=k)
    for i in range(len(docs)):
        print(docs[i].page_content)
'''
