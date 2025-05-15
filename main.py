import os
import yaml

from src.data_loader import data_load, data_process, data_chunking
from src.vector_db import generate_vector_db, load_vector_db


if __name__ == "__main__":
    try:
        # Config 불러오기
        with open("config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # 설정 확인
        verbose = config["settings"].get("verbose", False)
        if not isinstance(verbose, bool):
            raise ValueError("❌(config.settings.verbose) verbose는 True 또는 False여야 합니다.")
        if verbose:
            print("Verbose 모드로 실행 중입니다.")

        # HWP 파일을 PDF로 변환
        folder_path = os.path.abspath(config["data"]["folder_path"])
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"❌(config.data.folder_path) 폴더 경로가 존재하지 않습니다: {folder_path}")
        # batch_convert_hwp_to_pdf(folder_path)

        # 데이터 로드
        data_list_path = os.path.abspath(config["data"]["data_list_path"])
        if not os.path.exists(data_list_path):
            raise FileNotFoundError(f"❌(config.data.data_list_path) 파일 목록 경로가 존재하지 않습니다: {data_list_path}")
        limit = config["data"]["limit"] 
        if not isinstance(limit, int):
            raise ValueError("❌(config.data.limit) limit은 정수여야 합니다.")

        df = data_load(data_list_path, limit=limit)
        print("✅ 데이터 로드 완료")

        # 데이터 전처리
        apply_ocr = config["data"].get("apply_ocr", False)
        if not isinstance(apply_ocr, bool):
            raise ValueError("❌(config.data.apply_ocr) apply_ocr는 True 또는 False여야 합니다.")
        file_type = config["data"]["type"]
        if file_type not in ["hwp", "pdf", "all"]:
            raise ValueError("❌(config.data.type) file_type은 'hwp', 'pdf', 'all' 중 하나여야 합니다.")
        if verbose:
            print(f"파일 타입: {file_type}")
            print(f"OCR 적용 여부: {apply_ocr}")

        df = data_process(df, apply_ocr=apply_ocr, file_type=file_type)
        print("✅ 데이터 전처리 완료")

        # 청크 생성
        splitter_type = config["chunk"]["splitter"]
        if not isinstance(splitter_type, str) or not splitter_type:
            raise ValueError("❌(config.chunk.splitter) 청크 분할기 타입은 문자열이어야 합니다.")
        chunk_size = config["chunk"]["size"]
        if chunk_size <= 0 or not isinstance(chunk_size, int):
            raise ValueError("❌(config.chunk.size) 청크 크기는 1 이상의 정수여야 합니다.")
        chunk_overlap = config["chunk"]["overlap"]
        if chunk_overlap < 0 or not isinstance(chunk_overlap, int):
            raise ValueError("❌(config.chunk.overlap) 청크 오버랩은 0 이상의 정수여야 합니다.")
        if verbose:
            print(f"청크 크기: {chunk_size}")
            print(f"청크 오버랩: {chunk_overlap}")

        all_chunks = data_chunking(df=df, splitter_type=splitter_type, size=chunk_size, overlap=chunk_overlap)
        print("✅ 청크 생성 완료")        

        # 벡터 DB 생성
        embed_model = config["embedding"]["model"]
        if not isinstance(embed_model, str) or not embed_model:
            raise ValueError("❌(config.embedding.model) 임베딩 모델명이 문자열이어야 합니다.")

        embeddings = generate_vector_db(all_chunks, embed_model)
        print("✅ Vector DB 생성")

        # 벡터 DB 로드
        vector_db_path = config["embedding"]["vector_db_path"]
        if not os.path.exists(vector_db_path):
            raise FileNotFoundError(f"❌(config.embedding.vector_db_path) 벡터 DB 경로가 존재하지 않습니다: {vector_db_path}")
        
        vector_store = load_vector_db(vector_db_path, embed_model)
        print("✅ Vector DB 로드")

        # 유사도 검색
        query = config["query"]["question"]
        if not isinstance(query, str):
            raise ValueError("❌(config.query.question) query는 문자열이어야 합니다.")
        top_k = config["query"]["top_k"]
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("❌(config.query.top_k) top_k는 1 이상의 정수여야 합니다.")
        if verbose:
            print(f"유사도 검색 쿼리: {query}")
            print(f"유사도 검색 결과 개수: {top_k}")

        docs = vector_store.similarity_search(query, k=top_k)
        for i, doc in enumerate(docs, start=1):
            print(f"\n📄 유사 문서 {i}:\n{doc.page_content}")

    except FileNotFoundError as e:
        print(f"❌ [FileNotFound] \n {e}")

    except ValueError as e:
        print(f"❌ [Value] \n {e}")

    except TypeError as e:
        print(f"❌ [Type] \n {e}")

    except yaml.YAMLError as e:
        print(f"❌ [YAML] config.yaml 파싱 중 오류 발생: \n    {e}")

    except ImportError as e:
        print(f"❌ [Import] 모듈 임포트 실패: \n {e}")

    except Exception as e:
        print(f"❌ [Runtime] 예상치 못한 오류 발생: \n {e}")