# 2025-LLM-Project: RFP Summarizer & QA Chatbot

---

![Demo](assets/demo_white.gif)

> **RFPilot**은 사용자의 질문을 실시간으로 처리해 관련 제안서를 탐색하여 응답을 생성합니다. **RFPilot**과 함께 수백건의 RFP를 신속하게 처리하고, 컨설팅에 집중하세요!

## 1. 📌 프로젝트 개요

---

- **B2G 입찰지원 전문 컨설팅 스타트업 – 'RFPilot'**
- RFP 문서를 요약하고, 사용자 질문에 실시간으로 응답하는 챗봇 시스템

> **배경**: 매일 수백 건의 기업 및 정부 제안요청서(RFP)가 게시되는데, 각 요청서 당 수십 페이지가 넘는 문건을 모두 검토하는 것은 불가능합니다. 이러한 과정은 비효율적이며, 중요한 정보를 빠르게 파악하기 어렵습니다.  
> **목표**: 사용자의 질문에 실시간으로 응답하고, 관련 제안서를 탐색하여 요약 정보를 제공하는 챗봇을 개발하여 컨설턴트의 업무 효율을 향상시키고자 합니다.  
> **기대 효과**: RAG 시스템을 통해 중요한 정보를 신속하게 제공함으로써, 제안서 검토 시간을 단축하고 컨설팅 업무에 보다 집중할 수 있는 환경을 조성합니다.

### 기술스택

- **언어**: ![Python](https://img.shields.io/badge/Python-3776AB?style=plastic&logo=Python&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00?style=plastic&logo=jupyter&logoColor=white)
- **프레임워크**: ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=plastic&logo=Streamlit&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=plastic&logo=FastAPI&logoColor=white)
- **라이브러리**: ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=plastic&logo=PyTorch&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=plastic&logo=OpenAI&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=plastic&logo=HuggingFace&logoColor=black)
![FAISS](https://img.shields.io/badge/FAISS-00599C?style=plastic&logo=FAISS&logoColor=white)
![Chroma](https://img.shields.io/badge/Chroma-8E44AD?style=plastic&logo=Chroma&logoColor=white)
- **도구**: ![GitHub](https://img.shields.io/badge/GitHub-181717?style=plastic&logo=GitHub&logoColor=white)
![JupyterLab](https://img.shields.io/badge/JupyterLab-F37626?style=plastic&logo=Jupyter&logoColor=white)
![Notion](https://img.shields.io/badge/Notion-000000?style=plastic&logo=Notion&logoColor=white)
![Canva](https://img.shields.io/badge/Canva-00C4CC?style=plastic&logo=Canva&logoColor=white)
![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?style=plastic&logo=discord&logoColor=white)

## 2. ⚙️ 설치 및 실행 방법

---

```bash
# 1. 가상환경 설치
conda env create -f environment.yaml
conda activate myenv

# 2. 실행
chmod +x run.sh
./run.sh
```

## 3. 📂 프로젝트 구조

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
│   ├── embedding/           # 임베딩, 벡터DB 생성
│   ├── retriever/           # 문서 검색기
│   ├── generator/           # 응답 생성기
│   ├── streamlit/           # UI 구성
│   └── utils/               # 공통 함수 모듈
├── backend/                 # 백엔드
├── notebooks/               # 실험 및 테스트 노트북
├── run.sh                   # 실행 스크립트
└── README.md
```

- `main.py`: 전체 RAG 파이프라인 실행의 진입점입니다.
- `config.yaml`: 모델, 벡터DB, 경로 등 프로젝트 전반의 설정을 관리합니다.
- `environment.yaml`: 프로젝트 실행에 필요한 Conda 가상환경 설정 파일입니다.
- `data/`: 원문 문서, 생성된 벡터DB 등이 저장됩니다.
- `src/loader`: PDF, HWP 문서를 텍스트로 추출하고 의미 단위로 분할합니다.
- `src/embedding`: 텍스트 임베딩 벡터를 생성하고 FAISS/Chroma DB를 구축합니다.
- `src/retriever`: 사용자 질문에 대한 관련 문서를 벡터DB에서 검색합니다.
- `src/generator`: 검색된 문서 기반으로 LLM이 응답을 생성합니다.
- `src/streamlit`: Streamlit 기반 사용자 인터페이스를 구성합니다.
- `src/utils`: 설정 확인, 경로 설정 등 공통 유틸리티 함수들을 포함합니다.
- `backend/`: FastAPI 기반의 API 서버 코드가 포함되어 있습니다.
- `notebooks/`: 실험을 위한 Jupyter Notebook들이 포함됩니다.
- `run.sh`: 전체 시스템을 실행하기 위한 통합 셸 스크립트입니다.

## 4. 👥 팀 소개

---

> 인공지능 모델을 실제 공공 문서 분석에 적용해 실용적 도구를 만드는 것을 목표로 합니다.

### 멤버 소개

---

| 정영선 | 구극모 | 박규리 | 이학진 | 정재의 |
|:------:|:------:|:------:|:------:|:------:|
| <a href="https://github.com/YS-2357"><img src="https://github.com/YS-2357.png" width="100"/></a> | <a href="https://github.com/Glen0227"><img src="https://github.com/Glen0227.png" width="100"/></a> | <a href="https://github.com/gyurili"><img src="https://github.com/gyurili.png" width="100"/></a> | <a href="https://github.com/kyakyak"><img src="https://github.com/kyakyak.png" width="100"/></a> | <a href="https://github.com/JJU09"><img src="https://github.com/JJU09.png" width="100"/></a> |
| Def Programming<br>System Arch | Frontend Dev<br>Data Engineer | Backend Dev<br>RAG Engineer | NLP Engineer<br>Prompt Engineer | Frontend Dev<br>Data Processing |
| ![Gmail](https://img.shields.io/badge/joungyoungsun20@gmail.com-D14836?style=plastic&logo=gmail&logoColor=white) | ![Gmail](https://img.shields.io/badge/keugmo@gmail.com-D14836?style=plastic&logo=gmail&logoColor=white) | ![Naver](https://img.shields.io/badge/inglifestora@naver.com-03C75A?style=plastic&logo=naver&logoColor=white) | ![Gmail](https://img.shields.io/badge/udosjdjdjdj@gmail.com-D14836?style=plastic&logo=gmail&logoColor=white) | ![Gmail](https://img.shields.io/badge/jeaui54@gmail.com-D14836?style=plastic&logo=gmail&logoColor=white) |

## 5. 📊 타임라인

---

| 날짜 | 주요 내용 | 담당자 | 상태 |
| :---: | :--- | :---: | :---: |
| 2025-05-12 | 프로젝트 시작 및 팀 구성 | 전원 | 완료 |
| 2025-05-13 | 데이터 수집 및 전처리 시작 | 전원 | 완료 |
| 2025-05-14 | 데이터 로더 개발 | 이학진 | 완료 |
| 2025-05-15 | 임베딩 및 벡터DB 구현 | 구극모 | 완료 |
| 2025-05-15 | 리트리버 개발 및 실험 | 박규리 | 완료 |
| 2025-05-15 | 제너레이터 통합 및 구현 | 정재의 | 완료 |
| 2025-05-16 | 폴더구조 개편 및 main 통합 | 정영선 | 완료 |
| 2025-05-19 | Streamlit 기반 UI 제작 | 구극모, 정재의 | 완료 |
| 2025-05-20 | 전체 코드 기능 개선 및 수정 | 정영선, 박규리 | 완료 |
| 2025-05-21 | 프롬프트 엔지니어링 개선 | 이학진 | 완료 |
| 2025-05-22 | FastAPI 연동 실험 | 박규리 | 완료 |
| 2025-05-23 | 코드 리뷰 및 리팩토링 | 전원 | 완료 |
| 2025-05-27 | 모델 배포 실험 | 전원 | 완료 |
| 2025-05-28 | 보고서 및 발표자료 작성 | 전원 | 진행중 |
| 2025-06-02 | 프로젝트 최종 발표 및 배포 완료 | 전원 | 예정 |

## 6. 📎 참고 자료 및 산출물

---

- 📘 **최종 보고서**: [다운로드](https://drive.google.com/file/d/1y3Ksc8yg2JgfvVLnvcHZzNBn48v7iMOX/view?usp=sharing)
- 📽️ **발표자료 (PPT)**: [다운로드](https://your-link.com/presentation.pdf)
- 🗂️ **팀원별 협업 일지**
  - [정영선 협업일지](https://sapphire-cart-f52.notion.site/1f101c050cec803fb4aef0a5f8267fcf?pvs=74)
  - [구극모 협업일지](https://www.notion.so/1f1e1cd92be6809ba031d7caa012936e?source=copy_link)
  - [박규리 협업일지](https://www.notion.so/1f1caf59f0188065bec3c9fefc30f7e3?pvs=4)
  - [이학진 협업일지](https://www.notion.so/1f200f54e76e808e9a86f43a85d79afc?pvs=4)
  - [정재의 협업일지](https://www.notion.so/LLM-RAG-RFP-1f219af16ea580fd9603fc066bd71238?source=copy_link)

## 7. 📄 사용한 모델 및 라이센스

---

- **nlpai-lab/KoE5**: MIT License (상업적 사용 가능)
- **OpenAI text-embedding-3-large**: OpenAI API 전용 (상업적 사용 가능, API 기반)
- **Markr-AI/Gukbap-Qwen2.5-7B**: CC BY-NC 4.0 (비상업적 사용만 허용)
- **OpenAI GPT-4.1-nano**: OpenAI API 전용 (상업적 사용 가능, API 기반)