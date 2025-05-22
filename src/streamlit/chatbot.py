# 터미널 실행 코드
# python -m streamlit run src/streamlit/chatbot.py

# 외부 임포트
import os
import time 
import streamlit as st
import shutil
from pathlib import Path
from datetime import datetime
os.environ["HF_HOME"] = "2025-LLM-Project/.cache" # Huggingface 캐시 경로 설정

# 내부 임포트
from dotenv import load_dotenv
from src.utils.config import load_config
from src.loader.loader_main import loader_main
from src.utils.path import get_project_root_dir
from src.embedding.embedding_main import embedding_main
from src.retrieval.retrieval_main import retrieval_main
from src.generator.generator_main import generator_main
from src.embedding.embedding_main import generate_index_name

# Streamlit 페이지 설정
st.set_page_config(page_title="RFP Chatbot", layout="wide") # (Chrome 상단 바 이름)
st.header("RFP Chatbot", divider='blue') # 색상 선택 ("rainbow", "red", "blue", "green", "orange", "violet", "gray")
st.write("PDF, HWP 형식의 제안서를 업로드하여 내용 요약 및 질의응답을 경험하세요!")

# 세션 상태 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # [{"role": "user", "content": "..."}, {"role": "ai", "content": "..."}]

# 기본 설정 파일 경로
project_root = get_project_root_dir()
config_path = os.path.join(project_root, "config.yaml")
config = load_config(config_path)
dotenv_path = os.path.join(project_root, ".env")
load_dotenv(dotenv_path=dotenv_path)

# 사이드 바 설정
with st.sidebar:
    st.header("⚙️ 설정")

    # Data 관련 설정
    st.subheader("📂 데이터 설정")
    config["data"]["top_k"] = st.slider("🔢 최대 문서 수(files)", 1, 100, config["data"]["top_k"])
    config["data"]["file_type"] = st.selectbox("📄 파일 유형", ["all", "pdf", "hwp"], index=["all", "pdf", "hwp"].index(config["data"]["file_type"]))
    config["data"]["apply_ocr"] = st.toggle("🧾 OCR 적용 여부", config["data"]["apply_ocr"])
    config["data"]["splitter"] = st.selectbox("✂️ 문서 분할 방법", ["section", "recursive", "token"], index=["section", "recursive", "token"].index(config["data"]["splitter"]))
    config["data"]["chunk_size"] = st.number_input("📏 Chunk 크기", value=config["data"]["chunk_size"], step=100)
    config["data"]["chunk_overlap"] = st.number_input("🔁 Chunk 오버랩", value=config["data"]["chunk_overlap"], step=10)

    # Embedding 설정
    st.subheader("🧠 임베딩 설정")
    config["embedding"]["embed_model"] = st.text_input("🧬 임베딩 모델", config["embedding"]["embed_model"])
    config["embedding"]["db_type"] = st.selectbox("💾 Vector DB 타입", ["faiss", "chroma"], index=["faiss", "chroma"].index(config["embedding"]["db_type"]))

    if config["embedding"]["embed_model"].strip().lower() == "openai":
        load_dotenv()
        if not os.environ["OPENAI_API_KEY"]:
            openai_key = st.text_input("🔑 OpenAI API Key", type="password")
            os.environ["OPENAI_API_KEY"] = openai_key
            if not openai_key:
                st.warning("OpenAI 모델을 사용하려면 API 키를 입력해야 합니다.")

    # Retriever 설정
    st.subheader("🔍 리트리버 설정")
    config["retriever"]["search_type"] = st.selectbox("🔎 검색 방식", ["similarity", "hybrid"], index=["similarity", "hybrid"].index(config["retriever"]["search_type"]))
    config["retriever"]["top_k"] = st.slider("📄 검색 문서 수(chunks)", 1, 20, config["retriever"]["top_k"])
    config["retriever"]["rerank"] = st.toggle("📊 리랭크 적용", config["retriever"]["rerank"])
    config["retriever"]["rerank_top_k"] = st.slider("🔝 리랭크 문서 수(chunks)", 1, 20, config["retriever"]["rerank_top_k"])

    # Generator 설정
    st.subheader("🔍 생성자 설정")
    config["generator"]["model_type"] = st.selectbox("🔎 생성 모델 타입", ["huggingface", "openai"], index=["huggingface", "openai"].index(config["generator"]["model_type"]))
    config["generator"]["model_name"] = st.text_input("🧬 생성 모델", config["generator"]["model_name"])
    config["generator"]["max_length"] = st.number_input("🔢 최대 토큰 수(max_length)", value=config["generator"]["max_length"], step=32)

    if config["generator"]["model_type"].strip().lower() == "openai":
        load_dotenv()
        if not os.environ["OPENAI_API_KEY"]:
            openai_key = st.text_input("🔑 OpenAI API Key", type="password")
            os.environ["OPENAI_API_KEY"] = openai_key  # 필요한 경우 환경 변수로 설정
            if not openai_key:
                st.warning("OpenAI 모델을 사용하려면 API 키를 입력해야 합니다.")


    reset_vector_db = st.button("⚠️ Vector DB 초기화")
    
    if config["embedding"]["db_type"] == "faiss":
        faiss_index_name = f"{generate_index_name(config)}"
        vector_db_file = os.path.join(project_root, 'data', f"{faiss_index_name}.faiss")
        metadata_file = os.path.join(project_root, 'data', f"{faiss_index_name}.pkl")
    else:
        chroma_folder_name = f"{generate_index_name(config)}"
        chroma_path = os.path.join(project_root, 'data', chroma_folder_name)


    if reset_vector_db:
        # 선택된 벡터 DB 경로 삭제
        if config["embedding"]["db_type"] == "faiss":
            if os.path.exists(vector_db_file):
                os.remove(vector_db_file)
                os.remove(metadata_file)
                st.success("FAISS DB 삭제 완료")
        elif config["embedding"]["db_type"] == "chroma":
            import shutil
            if os.path.exists(chroma_path):
                shutil.rmtree(chroma_path)
                st.success("Chroma DB 삭제 완료")
        else:
            st.info("삭제할 파일 및 폴더가 없습니다.")

# 채팅
query = st.chat_input("질문을 입력하세요")

if query:
    # Vector DB 존재 여부 확인
    if config["data"]["top_k"] == 100:
        if config["embedding"]["db_type"] == "faiss":
            is_save = not os.path.exists(vector_db_file)
        elif config["embedding"]["db_type"] == "chroma":
            is_save = not os.path.exists(chroma_path)
        else:
            is_save = True
    else:
        is_save = True

    # 이전 대화로 context 구성
    st.session_state.chat_history.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    config["retriever"]["query"] = query

    # 데이터 처리
    chunks = loader_main(config)

    with st.spinner("📂 관련 문서 임베딩 중..."):
        vector_store = embedding_main(config, chunks, is_save=is_save)

    with st.spinner("🔍 관련 문서 검색 중..."):
        docs = retrieval_main(config, vector_store, chunks)

    # 이전 문맥을 전달하는 방식 (선택사항 - 모델 구현에 따라)
    config["chat_history"] = st.session_state.chat_history

    # 질문에 대한 답변 생성, 추론 시간 측정
    start_time = time.time()
    with st.spinner("🤖 답변 생성 중..."):
        answer = generator_main(docs, config) # generator_main 함수에 docs와 query를 전달
    end_time = time.time()
    elapsed = round(end_time - start_time, 2)

    # 추론 결과, 추론 시간 표시
    with st.chat_message("assistant"):
        st.markdown(answer)
        st.markdown(f"🕒 **추론 시간:** {elapsed}초")


    # 대화 기록 업데이트
    st.session_state.chat_history.append({"role": "ai", "content": answer}) # 답변 기록 

# 이전 대화 보여주기
# if st.session_state.chat_history:
#     st.markdown("### 대화 기록")
#     for turn in st.session_state.chat_history:
#         role = "🙋‍♂️ 사용자" if turn["role"] == "user" else "🤖 AI"
#         st.markdown(f"**{role}:** {turn['content']}")

# 이전 대화 보여주기(업데이트 버전)
if st.session_state.chat_history:
    for turn in st.session_state.chat_history:
        with st.chat_message("user" if turn["role"] == "user" else "assistant"):
            st.markdown(turn["content"])
