# 2025-LLM-Project 리드미

## 📄 프로젝트 개요

- 목적: PDF/HWP 형식의 제안요청서(RFP) 문서를 처리하여, 검색 기반 질의응답 시스템(RAG) 구현
- 기술 스택: LangChain, Streamlit, PyMuPDF, EasyOCR, FAISS/Chroma, OpenAI/HuggingFace

---

## 📁 폴더 구조

```bash
2025-LLM-Project/
├── .github/             # GitHub 관련 설정 및 워크플로우
├── data/
│   └── files/           # 원본 문서 데이터
├── notebooks/           # 실험/테스트용 Jupyter 노트북
├── src/                 # 핵심 소스 코드
│   ├── loader/          # 문서 로딩 및 전처리
│   ├── embedding/       # 임베딩 생성 및 벡터 DB 관리
│   ├── retrieval/       # 유사도 검색, 리트리버 구성
│   ├── generator/       # LLM 응답 생성 모듈
│   └── utils/           # 공통 유틸 함수
```

---

## ⚙️ 설정 항목 요약 (`config.yaml`)

### 📂 data
- `folder_path`: 원문 파일 경로 (`data/files`)
- `data_list_path`: 문서 목록 CSV 경로
- `top_k`: 최대 검색 문서 수 (1~100)
- `file_type`: 허용 문서 타입 (`hwp`, `pdf`, `all`)
- `apply_ocr`: OCR 적용 여부 (이미지 기반 문서)
- `splitter`: 문서 청크 분할 방식 (`recursive`, `token`, `section`)
- `chunk_size`: 청크 크기 (기본 1000)
- `chunk_overlap`: 청크 간 중첩 범위 (기본 250)

### 🔗 embedding
- `embed_model`: 사용 임베딩 모델 (`nlpai-lab/KoE5`, `openai` 등)
- `db_type`: 벡터 DB 유형 (`faiss`, `chroma`)
- `vector_db_path`: 벡터 DB 저장 경로 (기본 `data`)

### 🔍 retriever
- `search_type`: 검색 방식 (`similarity`, `hybrid`)
- `query`: 테스트용 기본 질의문
- `top_k`: 검색 결과 청크 수
- `rerank`: 재정렬 여부
- `min_chunks`: 문서별 최소 청크 보장 수

### 🤖 generator
- `model_type`: 생성 모델 소스 (`openai`, `huggingface`)
- `model_name`: 사용 모델 (`gpt-4.1-nano`, `Phi-4-mini-instruct` 등)
- `max_length`: 응답 최대 길이 (토큰 단위)
- `use_quantization`: 양자화 모델 사용 여부

### 🛠 settings
- `verbose`: 상세 로그 출력 여부 (True/False)

---

## 🧪 실험/테스트 환경 (Python 및 주요 라이브러리)

| 항목              | 버전          |
|-------------------|----------------|
| Python            | 3.10.12        |
| PyTorch           | 2.6.0 + cu124  |
| Transformers      | 4.51.3         |
| SentenceTransformers | 4.1.0      |
| FAISS             | cpu: 1.11.0, gpu: 1.7.2 |
| Streamlit         | 1.45.1         |
| LangChain         | 0.3.25         |
| HuggingFace Hub   | 0.31.1         |
| Scikit-Learn      | 1.6.1          |
| OpenAI            | 1.78.1         |

---

## 🛠️ 구현된 주요 기능 (기능별 진행 상황)

### ✅ Loader
- CSV 기반 문서 메타데이터 임베딩 및 유사 문서 검색 (메타 기반 Top-k)
- PDF 파일 OCR 기반 텍스트 추출 (EasyOCR + PyMuPDF)
- HWP 파일 처리 (HWPLoader)
- 청크 분할: Section 기반 + 길이 기준 병합 및 분할 (RecursiveCharacterTextSplitter)
- 청크 품질 분석 및 요약 출력

### ✅ Embedding
- OpenAI 및 HuggingFace 임베딩 모델 지원 (text-embedding-3-small, KoE5 등)
- FAISS 및 Chroma 벡터 DB 생성/저장/로드 지원
- embedding 모델 자동 감지 및 차원 설정
- DB 존재 여부 확인 후 자동 생성 or 로드 로직
- batch 단위 문서 임베딩 및 삽입

### ✅ Retriever
- similarity/hybrid 검색 타입 지원
- BM25 + Vector 기반 하이브리드 리트리버 구성
- 문서 내 유사도 정렬 및 선택 로직 구현
- 리랭크 적용 시 유사도 기반 정렬 유지 및 min/max 청크 수 보장

### ✅ Generatorㄲ
- 프롬프트 구성
  - 검색된 Document 리스트로부터 질문에 맞춘 입력 프롬프트 생성
  - 출처 정보(파일명, 기관, 사업명) 포함 가능하며, 커스텀 템플릿 사용 지원
- 모델 로딩 및 실행 (HuggingFace / OpenAI)
  - HuggingFace 모델: 정밀도(f16/4bit) 및 양자화 설정 지원, transformers 기반
  - OpenAI 모델: ChatCompletion API 사용, gpt-4.1-nano 등 지정 가능
- 응답 생성 및 후처리
  - 반복/존댓말/비정상적 응답 제거 필터 내장
  - LangSmith trace 기반 로깅 지원

---

> 본 리포지토리는 지속적으로 개선 중입니다. 코드/모듈별 사용법은 추후 상세 문서화 예정입니다.

---

## ▶️ Streamlit 실행 코드

```bash
streamlit run app.py --server.address=0.0.0.0 --server.port=8501
```

**Note:**
- 최대 문서 수를 `100`으로 지정하면 처음 한 번 전체 vector DB를 생성하고, 이후부터는 `load_vector_db()`를 통해 빠르게 작동합니다.  
  → **장점:** 첫 실행 시 약 5분 소요되며, 이후 쿼리 응답은 **1초 내외**로 처리됩니다.

- 최대 문서 수를 `100 미만`으로 지정하면 (예: `config['data']['top_k'] == 5`) 매 쿼리마다 vector DB를 **실시간 생성**하게 되며, 각 쿼리당 약 **30초 내외**가 소요됩니다.
