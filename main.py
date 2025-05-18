import os
import yaml

from src.data_loader import data_load, data_process, data_chunking
from src.vector_db import generate_vector_db, load_vector_db

# from src.retrieval import retrieve_documents


# if __name__ == "__main__":
#     try:
#         # Config 불러오기
#         with open("config.yaml", "r", encoding="utf-8") as f:
#             config = yaml.safe_load(f)

#         # 설정 Config
#         settings_config = config.get("settings", {})
#         if not isinstance(settings_config, dict):
#             raise ValueError("❌(config.settings) 설정은 딕셔너리여야 합니다.")
#         # verbose 설정
#         verbose = settings_config.get("verbose", False)
#         if not isinstance(verbose, bool):
#             raise ValueError("❌(config.settings.verbose) verbose는 True 또는 False여야 합니다.")
#         # verbose 출력
#         if verbose:
#             print("    -Verbose 모드로 실행 중입니다.")

#         # 데이터 Config
#         data_config = config.get("data", {})
#         if not isinstance(data_config, dict):
#             raise ValueError("❌(config.data) 데이터 설정은 딕셔너리여야 합니다.")
#         # HWP 파일을 PDF로 변환
#         folder_path = os.path.abspath(data_config.get("folder_path", ""))
#         if not isinstance(folder_path, str):
#             raise ValueError("❌(config.data.folder_path) 폴더 경로는 문자열이어야 합니다.")
#         if not os.path.exists(folder_path):
#             raise FileNotFoundError(f"❌(config.data.folder_path) 폴더 경로가 존재하지 않습니다: {folder_path}")
#         # batch_convert_hwp_to_pdf(folder_path)

#         # 데이터 로드
#         data_list_path = os.path.abspath(data_config.get("data_list_path", ""))
#         if not isinstance(data_list_path, str):
#             raise ValueError("❌(config.data.data_list_path) 데이터 목록 경로는 문자열이어야 합니다.")
#         if not os.path.exists(data_list_path):
#             raise FileNotFoundError(f"❌(config.data.data_list_path) 파일 목록 경로가 존재하지 않습니다: {data_list_path}")
#         limit = data_config.get("limit", 5)
#         if not isinstance(limit, int):
#             raise ValueError("❌(config.data.limit) limit은 정수여야 합니다.")
        
#         # verbose 출력
#         if verbose:
#             print(f"    -폴더 경로: {folder_path}")
#             print(f"    -데이터 로드 경로: {data_list_path}")
#             print(f"    -파일 수 제한: {limit}")

#         df = data_load(data_list_path, limit=limit)
#         print("✅ 데이터 로드 완료")

#         # 데이터 전처리
#         file_type = data_config.get("file_type", "all")
#         if file_type not in ["hwp", "pdf", "all"]:
#             raise ValueError("❌(config.data.type) file_type은 'hwp', 'pdf', 'all' 중 하나여야 합니다.")
#         apply_ocr = data_config.get("apply_ocr", False)
#         if not isinstance(apply_ocr, bool):
#             raise ValueError("❌(config.data.apply_ocr) apply_ocr는 True 또는 False여야 합니다.")
        
#         # verbose 출력
#         if verbose:
#             print(f"    -파일 타입: {file_type}")
#             print(f"    -OCR 적용 여부: {apply_ocr}")

#         df = data_process(df, apply_ocr=apply_ocr, file_type=file_type)
#         print("✅ 데이터 전처리 완료")

#         # 청크 Config
#         chunk_config = config.get("chunk", {})
#         if not isinstance(chunk_config, dict):
#             raise ValueError("❌(config.chunk) 청크 설정은 딕셔너리여야 합니다.")
        
#         # 청크 분할기 설정
#         splitter_type = chunk_config.get("splitter", "")
#         if not isinstance(splitter_type, str) or not splitter_type:
#             raise ValueError("❌(config.chunk.splitter) 청크 분할기 타입은 문자열이어야 합니다.")
#         chunk_size = chunk_config.get("size", 300)
#         if chunk_size <= 0 or not isinstance(chunk_size, int):
#             raise ValueError("❌(config.chunk.size) 청크 크기는 1 이상의 정수여야 합니다.")
#         chunk_overlap = chunk_config.get("overlap", 50)
#         if chunk_overlap < 0 or not isinstance(chunk_overlap, int):
#             raise ValueError("❌(config.chunk.overlap) 청크 오버랩은 0 이상의 정수여야 합니다.")
#         if verbose:
#             print(f"    -청크 분할기 타입: {splitter_type}")
#             print(f"    -청크 크기: {chunk_size}")
#             print(f"    -청크 오버랩: {chunk_overlap}")

#         all_chunks = data_chunking(df=df, splitter_type=splitter_type, size=chunk_size, overlap=chunk_overlap)
#         print("✅ 청크 생성 완료")        

#         # 벡터 DB Config
#         embed_config = config.get("embedding", {})
#         if not isinstance(embed_config, dict):
#             raise ValueError("❌(config.embedding) 벡터 DB 설정은 딕셔너리여야 합니다.")
#         index_name = generate_index_name(config)
        
#         # 벡터 DB 생성 또는 로드
#         embed_model = embed_config.get("model", "openai")
#         if not isinstance(embed_model, str) or not embed_model:
#             raise ValueError("❌(config.embedding.model) 임베딩 모델명이 문자열이어야 합니다.")
#         vector_db_path = embed_config.get("vector_db_path", "")
#         if not isinstance(vector_db_path, str):
#             raise ValueError("❌(config.embedding.vector_db_path) 벡터 DB 경로는 문자열이어야 합니다.")
#         if not os.path.exists(vector_db_path):
#             os.makedirs(vector_db_path, exist_ok=True)
#         db_type = embed_config.get("db_type", "faiss")
#         if db_type == "faiss":
#             faiss_file = os.path.join(vector_db_path, f"{index_name}.faiss")
#             pkl_file = os.path.join(vector_db_path, f"{index_name}.pkl")
#             db_exists = os.path.exists(faiss_file) and os.path.exists(pkl_file)
#         elif db_type == "chroma":
#             chroma_dir = os.path.join(vector_db_path, index_name)
#             db_exists = os.path.isdir(chroma_dir) and os.path.exists(os.path.join(chroma_dir, "chroma.sqlite3"))
#         else:
#             raise ValueError(f"❌(config.embedding.db_type) 지원하지 않는 DB 타입입니다: {db_type}")

#         # verbose 출력
#         if verbose:
#             print(f"    -임베딩 모델: {embed_model}")
#             print(f"    -DB 타입: {db_type}")
#             print(f"    -벡터 DB 경로: {vector_db_path}")
#             print(f"    -벡터 DB 파일: {index_name}")

#         if db_exists:
#             # 벡터 DB가 이미 존재하는 경우
#             if verbose:
#                 print(f"✅ 기존 벡터 DB 경로 발견됨: {index_name} 로드합니다.")
#             vector_store = load_vector_db(vector_db_path, embed_model, index_name, db_type)
#             print("✅ Vector DB 로드 완료")
#         else:
#             # 벡터 DB가 존재하지 않는 경우
#             if verbose:
#                 print(f"✅ 벡터 DB가 존재하지 않음. 새로 생성 후 저장합니다: {vector_db_path}")
#             vector_store = generate_vector_db(all_chunks, embed_model, index_name, db_type)
#             print("✅ Vector DB 생성 완료")

#         # 쿼리 Config
#         query_config = config.get("query", {})
#         if not isinstance(query_config, dict):
#             raise ValueError("❌(config.query) 쿼리 설정은 딕셔너리여야 합니다.")
        
#         # 유사도 검색
#         query = query_config.get("question", "")
#         if not isinstance(query, str):
#             raise ValueError("❌(config.query.question) query는 문자열이어야 합니다.")
#         top_k = query_config.get("top_k", 5)
#         if not isinstance(top_k, int) or top_k <= 0:
#             raise ValueError("❌(config.query.top_k) top_k는 1 이상의 정수여야 합니다.")
#         if verbose:
#             print(f"    -유사도 검색 쿼리: {query}")
#             print(f"    -유사도 검색 결과 개수: {top_k}")
        
#         search_type = config.get("retrieval", {}).get("search_type", "similarity")
#         if not isinstance(search_type, str):
#             raise ValueError("❌(config.retrieval.method) 검색 방식은 문자열이어야 합니다.")
        
#         docs = retrieve_documents(query, vector_store, top_k, search_type, all_chunks)
#         for i, doc in enumerate(docs, 1):
#             print(f"\n📄 문서 {i}")
#             print(f"본문:\n{doc['text'][:300]}...")
#             print(f"메타데이터: {doc['metadata']}")


#     # 예외 처리
#     except FileNotFoundError as e:
#         print(f"❌ [FileNotFound] \n  {e}")

#     except ValueError as e:
#         print(f"❌ [Value] \n  {e}")

#     except TypeError as e:
#         print(f"❌ [Type] \n  {e}")

#     except yaml.YAMLError as e:
#         print(f"❌ [YAML] config.yaml 파싱 중 오류 발생: \n  {e}")

#     except ImportError as e:
#         print(f"❌ [Import] 모듈 임포트 실패: \n  {e}")

#     except Exception as e:
#         print(f"❌ [Runtime] 예상치 못한 오류 발생: \n  {e}")