# 터미널 실행 코드
# python -m streamlit run src/streamlit/chatbot.py

# 외부 임포트
import os
import shutil
import yaml
import uuid
import torch
import streamlit as st
from typing import Dict
from dotenv import load_dotenv

# 내부 임포트
from src.utils.config import load_config
from src.utils.path import get_project_root_dir
from src.utils.shared_cache import set_cache_dirs
from src.embedding.embedding_main import generate_index_name
from src.generator.hf_generator import load_hf_model
from src.generator.openai_generator import load_openai_model
from src.embedding.vector_db import generate_embedding
from src.generator.chat_history import load_chat_history
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipe_main import rag_pipeline

set_cache_dirs()
load_dotenv()
torch.classes.__path__ = []
FASTAPI_URL = os.getenv("FASTAPI_URL")

# Streamlit 페이지 설정
st.set_page_config(
    page_title="2025-LLM-Project: RFP Summarizer & QA Chatbot", 
    layout="wide"
)

st.header("RFPilot", divider='blue')
st.caption("PDF, HWP 형식의 제안서를 기반으로 한 내용 요약 및 질의응답을 경험하세요!")

# 프로젝트 루트 경로 설정 및 config 로드
try:
    project_root = get_project_root_dir()
    config = load_config(project_root)
except Exception as e:
    st.error(f"❌ 설정 파일 로드 실패: {e}")
    st.stop()

# .env 파일 로딩 (API Key 등 private 정보 처리용)
dotenv_path = os.path.join(project_root, ".env")
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
else:
    st.warning(".env 파일을 찾을 수 없습니다. 일부 기능이 제한될 수 있습니다.")

# 세션 상태 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
else: # 세션 상태가 존재하는 경우, chat_history를 초기화하지 않음
    st.session_state.chat_history = st.session_state.get("chat_history", [])
    config["chat_history"] = st.session_state.chat_history

if "docs" not in st.session_state:
    st.session_state.docs = None

if "session_id" not in st.session_state:
    # uuid 고유 번호는 -(hyphen)을 포함해 36자, 너무 긺으로 자르는 과정 추가
    st.session_state.session_id = str(uuid.uuid4())[:8]

session_id = st.session_state.session_id

# 모델 불러오기 캐시 함수
@st.cache_resource
def get_generation_model(model_type: str, model_name: str, use_quantization: bool = False) -> Dict:
    """
    지정된 모델 타입 및 이름에 따라 생성 모델을 로드합니다.

    Args:
        model_type (str): 생성 모델 종류 ('huggingface' 또는 'openai')
        model_name (str): 사용할 모델 이름
        use_quantization (bool, optional): 양자화 사용 여부. 기본값은 False.

    Returns:
        Dict: 로드된 모델 정보 (예: pipeline, tokenizer 등 포함)
    """
    try:
        config = {
            'generator': {
                'model_type': model_type,
                'model_name': model_name,
                'use_quantization': use_quantization
            }
        }

        if model_type == 'huggingface':
            return load_hf_model(config)
        elif model_type == 'openai':
            return load_openai_model(config)
        else:
            raise ValueError(f"지원되지 않는 모델 타입: {model_type}")
    
    except Exception as e:
        st.error(f"모델 로딩 실패: {e}")
        st.stop()


def api_key_verification(embed_model):
    if embed_model.strip().lower() == "openai":
        load_dotenv()
        openai_key = os.environ.get("OPENAI_API_KEY")

        if not openai_key:
            openai_key = st.text_input(
                "🔑 OpenAI API Key",
                type="password",
                key=f"openai_api_key_special"
            )
            if openai_key:
                os.environ["OPENAI_API_KEY"] = openai_key
            else:
                st.warning("OpenAI 모델을 사용하려면 API 키를 입력해야 합니다.")



# 사이드바 구성
with st.sidebar:
    st.subheader("⚙️ 설정")
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

    # api_key 확인
    api_key_verification(config["embedding"]["embed_model"])

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

    # api_key 재확인
    api_key_verification(config["generator"]["model_type"])

    reset_vector_db = st.button("⚠️ Vector DB 초기화")
    
    if config["embedding"]["db_type"] == "faiss":
        faiss_index_name = f"{generate_index_name(config)}"
        vector_db_file = os.path.join(project_root, config['embedding']['vector_db_path'], f"{faiss_index_name}_{session_id}.faiss")
        metadata_file = os.path.join(project_root, config['embedding']['vector_db_path'], f"{faiss_index_name}_{session_id}.pkl")
    else:
        chroma_folder_name = f"{generate_index_name(config)}_{session_id}"
        chroma_path = os.path.join(project_root, config['embedding']['vector_db_path'], chroma_folder_name)


    if reset_vector_db:
        # 선택된 벡터 DB 경로 삭제
        try:
            if config["embedding"]["db_type"] == "faiss":
                if os.path.exists(vector_db_file):
                    os.remove(vector_db_file)
                    os.remove(metadata_file)
                    st.success("FAISS DB 삭제 완료")
                else:
                    st.info("FAISS 파일이 존재하지 않습니다.")
            elif config["embedding"]["db_type"] == "chroma":
                import shutil
                if os.path.exists(chroma_path):
                    shutil.rmtree(chroma_path)
                    st.success("Chroma DB 삭제 완료")
                else:
                    st.info("Chroma 폴더가 존재하지 않습니다.")
        except Exception as e:
            st.error(f"Vector DB 삭제 실패: {e}")
            
    # 설정 버튼
    cols = st.columns([4, 6])
    with cols[0]:
        if st.button("🔄 리셋"):
            st.session_state.chat_history = []
            st.session_state.docs = None
            st.session_state.past_chunks = []
            st.rerun()
    with cols[1]:
        if st.button("🔁 모델 리로드"):
            get_generation_model.clear()
            st.rerun()

# 탭 구성
tab1, tab2 = st.tabs(["💬 챗봇", "📄 문서 요약 및 분석"])

model_type = config["generator"]["model_type"]
model_name = config["generator"]["model_name"]
use_quantization = config["generator"]["use_quantization"]

print("\n📄 [Verbose] 최종 설정 내용:")
print(yaml.dump(config, allow_unicode=True, sort_keys=False))

with tab1:
    query = st.chat_input("질문을 입력하세요")

    # 질문 처리
    if query:
        if not isinstance(query, str) or query.strip() == "":
            st.warning("질문을 올바르게 입력해주세요.")
            st.stop()

        # 사이드바 설정 반영 - Vector DB 존재 여부 확인
        if config["data"]["top_k"] == 100:
            if config["embedding"]["db_type"] == "faiss":
                is_save = not os.path.exists(vector_db_file)
            elif config["embedding"]["db_type"] == "chroma":
                is_save = not os.path.exists(chroma_path)
            else:
                is_save = True
        else:
            is_save = True
            
        # 질문 입력시 이전 추출문서 기록 초기화
        if st.session_state.docs is not None:
            st.session_state.docs = None
        
        # 모델 불러오기는 단 한번만!
        model_info = get_generation_model(model_type, 
                                        model_name, 
                                        use_quantization)

        chat_history = load_chat_history(config, model_info)
        
        with st.chat_message("user"):
            st.markdown(query)

        try:
            with st.spinner("🤖 답변 생성 중..."):
                embeddings = generate_embedding(config["embedding"]["embed_model"])
                docs, answer, elapsed = rag_pipeline(config, embeddings, chat_history, model_info=model_info, is_save=is_save, session_id=session_id)
                
            # 결과 Streamlit에 반영
            st.session_state.docs = docs 
            
            # 대화 이력 업데이트
            st.session_state.chat_history.append({"role": "user", "content": query})
            st.session_state.chat_history.append({"role": "ai", "content": answer})
            
            # 대화 기록 업데이트
            config["chat_history"] = st.session_state.chat_history
            
            # 추론 시간 표시
            with st.chat_message("assistant"):
                st.markdown(f"🕒 **추론 시간:** {elapsed}초")
                
            # 랜더링 한계점: 20개까지 히스토리 표시
            MAX_CHAT_HISTORY = 20
            if len(st.session_state.chat_history) > MAX_CHAT_HISTORY:
                st.session_state.chat_history = st.session_state.chat_history[-MAX_CHAT_HISTORY:]

        except Exception as e:
            st.error(f"❌ 문서 처리 중 오류 발생: {e}")
            st.stop()

    # 이전 대화 출력
    for turn in st.session_state.chat_history[::-1]:
        with st.chat_message("user" if turn["role"] == "user" else "assistant"):
            st.markdown(turn["content"])

with tab2:
    st.subheader("📄 문서 요약 및 분석")

    docs = st.session_state.get("docs", None)

    if docs is None:
        st.info("❗ 먼저 질문을 입력하고 문서를 검색하세요.")
    elif isinstance(docs, list) and len(docs) > 0:
        for i, doc in enumerate(docs):
            with st.expander(f"[{i+1}] {doc['metadata'].get('사업명', '제목 없음')}"):
                st.write("📄 **메타데이터**")
                st.json(doc["metadata"])
                st.write("📝 **문서 내용**")
                st.write(doc["content"])
    elif isinstance(docs, list) and len(docs) == 0:
        st.warning("검색된 문서가 없습니다.")
    else:
        st.info(docs.page_content)