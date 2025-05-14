import os
import yaml

from src.data_load import data_load, data_process, data_chunking
from src.vector_db import generate_vector_db, load_vector_db


if __name__ == "__main__":
    try:
        # Config 불러오기
        with open("config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # 설정 확인
        if config["settings"]["verbose"]:
            print("Verbose 모드로 실행 중입니다.")
            print(config)

        # HWP 파일을 PDF로 변환
        folder_path = os.path.abspath(config["data"]["folder_path"])
        # batch_convert_hwp_to_pdf(folder_path)

        # PDF 또는 HWP 파일에서 텍스트 추출
        data_list_path = os.path.abspath(config["data"]["data_list_path"])

        # 데이터 로드
        df = data_load(data_list_path)
        print("✅ 데이터 로드 완료")

        # 데이터 전처리 및 청크 생성
        apply_ocr = config["data"]["apply_ocr"]
        file_type = config["data"]["type"]
        if file_type not in ["hwp", "pdf", "all"]:
            raise ValueError("file_type은 'hwp', 'pdf', 'all' 중 하나여야 합니다.")
        
        if config["settings"]["verbose"]:
            print(f"파일 타입: {file_type}")
            print(f"OCR 적용 여부: {apply_ocr}")

        df = data_process(df, apply_ocr=apply_ocr, file_type=file_type)
        print("✅ 데이터 전처리 완료")

        all_chunks = data_chunking(df)
        print("✅ 청크 생성 완료")        

        # 벡터 DB 생성
        embed_model_name = config["embedding"]["model"]
        #❌ 나중에 방어적 코드 추가해야함


        embeddings = generate_vector_db(all_chunks, embed_model_name)
        print("✅ Vector DB 생성")

        # 벡터 DB 로드
        vector_db_path = config["embedding"]["vector_db_path"]
        embed_model = config["embedding"]["model"]
        if not os.path.exists(vector_db_path):
            raise FileNotFoundError(f"벡터 DB 경로가 존재하지 않습니다: {vector_db_path}")
        
        vector_store = load_vector_db(vector_db_path, embed_model)
        print("✅ Vector DB 로드")

        # 유사도 검색
        query = config["query"]["text"]
        k = config["query"]["top_k"]
        if k <= 0:
            raise ValueError("k는 1 이상의 정수여야 합니다.")
        if config["settings"]["verbose"]:
            print(f"유사도 검색 쿼리: {query}")
            print(f"유사도 검색 결과 개수: {k}")

        docs = vector_store.similarity_search(query, k=k)
        for i, doc in enumerate(docs, start=1):
            print(f"\n📄 유사 문서 {i}:\n{doc.page_content}")

    except Exception as e:
        print(f"❌ Error: {e}")