# 2025-LLM-Project: RFP Summarizer & QA Chatbot

---

## 시연영상 들어가야함

> 입찰메이트 봇은 사용자의 질문을 실시간으로 처리해 관련 제안서를 탐색하여 응답을 생성합니다. 입찰메이트 봇과 함께 수백건의 RFP를 신속하게 처리하고, 컨설팅이 집중하세요!
>

- **언어**: ![Python](https://img.shields.io/badge/Python-3776AB?style=plastic&logo=Python&logoColor=white)
- **프레임워크**: ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=plastic&logo=Streamlit&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=plastic&logo=FastAPI&logoColor=white)
- **라이브러리**: ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=plastic&logo=PyTorch&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=plastic&logo=OpenAI&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=plastic&logo=HuggingFace&logoColor=black)
![FAISS](https://img.shields.io/badge/FAISS-00599C?style=plastic&logo=FAISS&logoColor=white)
![Chroma](https://img.shields.io/badge/Chroma-8E44AD?style=plastic&logo=Chroma&logoColor=white)
- **도구**: ![GitHub](https://img.shields.io/badge/GitHub-181717?style=plastic&logo=GitHub&logoColor=white)
![Notion](https://img.shields.io/badge/Notion-000000?style=plastic&logo=Notion&logoColor=white)
![Canva](https://img.shields.io/badge/Canva-00C4CC?style=plastic&logo=Canva&logoColor=white)
- **미정**: ![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00?style=plastic&logo=jupyter&logoColor=white)


## 1. 📌 프로젝트 개요

---

- **B2G 입찰지원 전문 컨설팅 스타트업 – ‘입찰메이트 봇’**

    > **배경**: 매일 수백 건의 RFP가 게시되는데, 각 요청서 당 수십 페이지가 넘는 문건을 모두 검토하는 것은 불가능합니다. 이러한 과정은 비효율적이며, 중요한 정보를 빠르게 파악하기 어렵습니다.
    **목표**: 사용자의 질문에 실시간으로 응답하고, 관련 제안서를 탐색하여 요약 정보를 제공하는 챗봇을 개발하여 컨설턴트의 업무 효율을 향상시키고자 합니다.
    **기대 효과**: RAG 시스템을 통해 중요한 정보를 신속하게 제공함으로써, 제안서 검토 시간을 단축하고 컨설팅 업무에 보다 집중할 수 있는 환경을 조성합니다.

## 3. ⚙️ 설치 및 실행 방법

---

```bash
# 1. 의존성 설치
pip install -r requirements.txt
conda activate ?

# 2. 실행
python -m streamlit run src/streamlit/chatbot.py
```

## 4. 📂 프로젝트 구조

---

```arduino
2025-LLM-Project/
│
├── main.py                  # 실행 진입점
├── config.yaml              # 설정 파일
├── environment.yaml         # conda 환경 파일
├── data/                    # 문서 및 벡터DB 저장 폴더
├── src/
│   ├── loader/              # 문서 로딩 및 전처리
│   ├── embedding/           # 임베딩 생성
│   ├── retriever/           # 문서 검색기
│   ├── generator/           # 응답 생성기
│   ├── streamlit/           # UI 구성
│   └── utils/               # 공통 함수 모듈
├── backend/                 # 백엔드
├── notebooks/               # 실험 및 테스트 노트북
├── run.sh                   # 실행 스크립트
└── README.md
```

### 📁 각 디렉토리 설명

- `main.py`: 전체 RAG 파이프라인을 실행하는 엔트리포인트
- `src/loader`: PDF/HWP 등의 문서를 텍스트로 로딩하고 의미 단위로 분할합니다.
- `src/embedding`: 텍스트에 대한 임베딩 벡터를 생성 벡터DB를 만듭니다.
- `src/retriever`: 사용자 질문에 관련된 문서를 벡터DB에서 검색합니다.
- `src/generator`: 검색된 문서 기반으로 LLM이 응답을 생성합니다.
- `src/streamlit`: 사용자 인터페이스를 구성하며 FastAPI와 연동합니다.
- `config.yaml`: 모델, DB, 경로 등 전반적인 설정을 관리합니다.
- `data/`: 원문 문서와 생성된 임베딩 데이터가 저장됩니다.
- `notebooks/`: 개발 과정 중의 실험 및 테스트 코드 기록입니다.
- `run.sh`: 실행 자동화를 위한 셸 스크립트

## 6. 👥 팀 소개

---

### 팀 소개

---

> 인공지능 모델을 실제 공공 문서 분석에 적용해 실용적 도구를 만드는 것을 목표로 합니다.

### 멤버 소개

| 정영선 | 구극모 | 박규리 | 이학진 | 정재의 |
|--------|----------|--------|--------|--------|
| <img src="https://github.com/YS-2357.png" width="100"/> | <img src="https://github.com/Glen0227.png" width="100"/> | <img src="https://github.com/gyurili.png" width="100"/> | <img src="https://github.com/kyakyak.png" width="100"/> | <img src="https://github.com/JJU09.png" width="100"/> |
| PM  | 프론트엔드 개발자 | 백엔드 엔지니어 | NLP 엔지니어 | 문서 처리 및 전처리|
| 그외? | | | | |