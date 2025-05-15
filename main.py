import os
import yaml

from src.data_loader import data_load, data_process, data_chunking
from src.vector_db import generate_vector_db, load_vector_db
from src.retriever import retrieve_documents
from src.generation import load_chat_model, build_qa_chain


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

        # 데이터 로드
        data_list_path = os.path.abspath(config["data"]["data_list_path"])
        if not os.path.exists(data_list_path):
            raise FileNotFoundError(f"❌ 파일 목록 경로가 존재하지 않습니다: {data_list_path}")
        limit = config["data"]["limit"]
        if limit <= 0 or limit > 100 or not isinstance(limit, int):
            raise ValueError("❌ limit은 1 이상 100 이하의 정수여야 합니다.")

        df = data_load(data_list_path, limit=limit)
        print("✅ 데이터 로드 완료")

        # 데이터 전처리
        apply_ocr = config["data"]["apply_ocr"]
        if not isinstance(apply_ocr, bool):
            raise ValueError("❌ apply_ocr는 True 또는 False여야 합니다.")
        file_type = config["data"]["type"]
        if file_type not in ["hwp", "pdf", "all"]:
            raise ValueError("❌ file_type은 'hwp', 'pdf', 'all' 중 하나여야 합니다.")
        if config["settings"]["verbose"]:
            print(f"파일 타입: {file_type}")
            print(f"OCR 적용 여부: {apply_ocr}")

        df = data_process(df, apply_ocr=apply_ocr, file_type=file_type)
        print("✅ 데이터 전처리 완료")

        # 청크 생성
        splitter_type = config["chunk"]["splitter"]
        if splitter_type not in ["recursive", "token"]:
            raise ValueError("❌ spliter_type은'recursive' 또는 'token' 중 하나여야 합니다.")
        chunk_size = config["chunk"]["size"]
        if chunk_size <= 0 or not isinstance(chunk_size, int):
            raise ValueError("❌ 청크 크기는 1 이상의 정수여야 합니다.")
        chunk_overlap = config["chunk"]["overlap"]
        if chunk_overlap < 0 or not isinstance(chunk_overlap, int):
            raise ValueError("❌ 청크 오버랩은 0 이상의 정수여야 합니다.")
        if config["settings"]["verbose"]:
            print(f"청크 크기: {chunk_size}")
            print(f"청크 오버랩: {chunk_overlap}")

        all_chunks = data_chunking(df=df, splitter_type=splitter_type, size=chunk_size, overlap=chunk_overlap)
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
        query = config["query"]["question"]
        k = config["query"]["top_k"]
        search_type = config["retrieval"]["method"]

        if k <= 0:
            raise ValueError("k는 1 이상의 정수여야 합니다.")
        if config["settings"]["verbose"]:
            print(f"유사도 검색 쿼리: {query}")
            print(f"검색 방식: {search_type}, 개수: {k}")
            
        docs = retrieve_documents(query, vector_store, k, search_type)
        for i, doc in enumerate(docs, 1):
            print(f"\n📄 문서 {i}")
            print(f"본문:\n{doc['text'][:300]}...")
            print(f"메타데이터: {doc['metadata']}")
        
        '''
        query = config["query"]["question"]
        k = config["query"]["top_k"]
        if k <= 0:
            raise ValueError("k는 1 이상의 정수여야 합니다.")
        if config["settings"]["verbose"]:
            print(f"유사도 검색 쿼리: {query}")
            print(f"유사도 검색 결과 개수: {k}")

        docs = vector_store.similarity_search(query, k=k)
        for i, doc in enumerate(docs, start=1):
            print(f"\n📄 유사 문서 {i}:\n{doc.page_content}")
            
        
        # LLM 모델 로드
        model_config = config["model"]
        query_config = config["query"]
        chat_model = load_chat_model(model_config)
        print("✅ 생성형 LLM 로드")

        # QA 체인 구성
        qa_chain = build_qa_chain(
            vector_store=vector_store,
            chat_model=chat_model,
            top_k=query_config["top_k"]
        )

        query = query_config["question"]
        response = qa_chain.invoke(query)

        print(f"\n💬 LLM 응답:\n{response}")
        '''

    except Exception as e:
        print(f"❌ Error: {e}")