import os
import shutil
import streamlit as st
from dotenv import load_dotenv

from src.utils.path import get_project_root_dir
from src.utils.config import load_config
from src.loader.loader_main import loader_main
from src.embedding.embedding_main import embedding_main
from src.retrieval.retrieval_main import retrieval_main
from src.generator.generator_main import generator_main, generate_with_clarification
from src.embedding.embedding_main import generate_index_name
from src.generator.hf_generator import load_hf_model
from src.generator.openai_generator import load_openai_model
from src.generator.load_model import load_generator_model


project_root = get_project_root_dir()
config_path = os.path.join(project_root, "config.yaml")
config = load_config(config_path)

st.set_page_config(page_title="RAG Demo", layout="wide")
st.title("🧠 질문 기반 문서 검색 및 답변 생성")

# ======================
# 🔧 설정 옵션 UI
# ======================

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

@st.cache_resource
def get_generation_model(model_type:str, model_name:str, use_quantization:bool = False):
    config = {'generator': {'model_type': model_type, 'model_name': model_name, 'use_quantization': use_quantization}}
    if model_type == 'huggingface':
        model_info = load_hf_model(config)
    else:
        model_info = load_openai_model(config)
    return model_info
    
def run_rag_pipeline(config):
    '''
    '''
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
    # 문서 로드 및 임베딩
    chunks = loader_main(config) if is_save else []

    with st.spinner("📂 관련 문서 임베딩 중..."):
        vector_store = embedding_main(config, chunks, is_save=is_save)

    # 문서 검색
    with st.spinner("🔍 관련 문서 검색 중..."):
        docs = retrieval_main(config, vector_store, chunks)

    # 검색된 문서 출력
    st.markdown("### 🔎 검색된 문서 정보")
    if isinstance(docs, list) and len(docs) > 0:
        for i, doc in enumerate(docs):
            with st.expander(f"[{i+1}] {doc.metadata.get('사업명', '알 수 없는 문서')}"):
                st.write("📄 **메타데이터**")
                st.json(doc.metadata)
                st.write("📝 **문서 chunk 내용**")
                st.write(doc.page_content)
    elif isinstance(docs, list) and len(docs) == 0:
        st.warning("검색된 문서가 없습니다.")
    else:
        st.info(docs.page_content)
    model_type = config["generator"]["model_type"]
    model_name = config["generator"]["model_name"]
    use_quantization = config["generator"]["use_quantization"]
    model_info = get_generation_model(model_type, model_name, use_quantization)
    
    # 답변 생성
    with st.spinner("문서 요약 및 답변 생성 중..."):
        answer = generate_with_clarification(docs, config, max_retries=3, model_info=model_info)

    # 답변 표시
    st.markdown("### 🤖 요약된 답변")
    st.success(answer)

    # 대화 히스토리 저장
    st.session_state.chat_history.append({
        "question": config["retriever"]["query"],
        "answer": answer
    })

# ======================
# 🤖 질문 입력 및 실행
# ======================
if "input_key_version" not in st.session_state:
    st.session_state.input_key_version = 0
if "trigger_search" not in st.session_state:
    st.session_state.trigger_search = False
if "user_query" not in st.session_state:
    st.session_state.user_query = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def reset_query():
    st.session_state.input_key_version += 1
    st.session_state.user_query = "" 
    st.session_state.trigger_search = False

# 항상 질문 입력창 보여줌
query_key = f"user_query_{st.session_state.input_key_version}"
query = st.text_input("❓ 질문을 입력하세요:", key=query_key)

if st.button("🔎 검색") and query.strip():
    st.session_state.trigger_search = True
    st.session_state.user_query = query

if st.session_state.trigger_search:
    config["retriever"]["query"] = st.session_state.user_query
    st.markdown(f"### 🙋 입력한 질문: `{st.session_state.user_query}`")

    # RAG 실행
    run_rag_pipeline(config)
    # Reset
    reset_query()
