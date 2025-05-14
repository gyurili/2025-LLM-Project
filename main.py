import os
import yaml

from src.data_load import data_load, data_process, hwp_chunking, generate_vector_db, load_vector_db
from src.hwp_to_pdf import batch_convert_hwp_to_pdf
from src.pdf_loader import process_all_pdfs_in_folder


if __name__ == "__main__":
    # Config 불러오기
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config["settings"]["verbose"]:
        print("Verbose 모드로 실행 중입니다.")
        print(config)

    # HWP 파일을 PDF로 변환
    folder_path = os.path.abspath(config["data"]["folder_path"])
    # batch_convert_hwp_to_pdf(folder_path)

    # PDF 또는 HWP 파일에서 텍스트 추출
    data_list_path = os.path.abspath(config["data"]["data_list_path"])
    df = data_load(data_list_path)

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
