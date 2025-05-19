import os
import fitz
import easyocr
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from typing import List
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_teddynote.document_loaders import HWPLoader
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.utils.path import get_project_root_dir

# EasyOCR Reader 객체 생성 (GPU 사용)
# 한글(ko) + 영어(en)를 인식하며, 모델은 한 번만 로드됨
reader = easyocr.Reader(['ko', 'en'], gpu=True)


def safe_ocr(img_array: np.ndarray, ocr_reader: easyocr.Reader) -> str:
    """
    이미지 배열을 입력받아 EasyOCR로 텍스트를 추출합니다.

    Args:
        img_array (np.ndarray): OCR을 수행할 이미지 배열
        ocr_reader (easyocr.Reader): 초기화된 EasyOCR 리더 인스턴스

    Returns:
        str: 추출된 텍스트 문자열
    """
    try:
        result = ocr_reader.readtext(img_array, detail=0)
        if not isinstance(result, list):
            return ""
        return "\n".join(result)
    except Exception as e:
        raise RuntimeError(f"❌ [OCR] (data_loader.safe_ocr) OCR 처리 실패: {e}")


def extract_text_from_pdf(pdf_path: Path, apply_ocr: bool = True) -> str:
    """
    PDF 파일에서 텍스트 및 OCR 텍스트를 추출합니다.

    Args:
        pdf_path (Path): 처리할 PDF 파일 경로
        apply_ocr (bool): OCR 수행 여부

    Returns:
        str: 전체 페이지에서 추출된 텍스트
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"❌(data_loader.extract_text_from_pdf.pdf_path) PDF 파일을 찾을 수 없습니다: {pdf_path}")
    full_text = ""
    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(tqdm(doc, desc=f"{pdf_path.name}")):
            page_text = page.get_text()
            full_text += page_text

            if apply_ocr:
                pix = page.get_pixmap(dpi=300)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                ocr_text = safe_ocr(np.array(img), reader)
                if ocr_text.strip():
                    full_text += f"\n[OCR p.{page_num + 1}]\n{ocr_text}"
    return full_text


def retrieve_top_documents_from_metadata(query, csv_path, top_k=5, verbose=False):
    """
    사용자 질문(query)과 문서 메타데이터(csv)에 기반하여 
    가장 유사한 top_k개의 문서를 반환합니다.

    Parameters:
        query (str): 사용자 질문
        csv_path (str): CSV 파일 경로
        top_k (int): 반환할 문서 수 (기본값 5)

    Returns:
        pd.DataFrame: 상위 top_k 문서 정보 + 유사도 점수
    """
    try:
        # 0. 모델 로드
        sbert_model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
    except Exception as e:
        raise RuntimeError(f"❌ SBERT 모델 로딩 실패: {str(e)}")

    # 1. CSV 파일 로드
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ 파일을 찾을 수 없습니다: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"❌ CSV 파일 로딩 실패: {str(e)}")
    
    # 2. 필요한 열 존재 여부 확인
    required_columns = ["사업명", "발주 기관", "사업 요약", "파일명"]
    for col in required_columns:
        if col not in df.columns:
            raise KeyError(f"❌ '{col}' 열이 CSV에 존재하지 않습니다.")

    # 3. 임베딩용 텍스트 생성
    def make_embedding_text(row):
        return f"{row['사업명']} {row['발주 기관']} {row['사업 요약']}"
    
    try:
        df["임베딩텍스트"] = df.apply(make_embedding_text, axis=1)
    except Exception as e:
        raise RuntimeError(f"❌ 임베딩 텍스트 생성 중 오류: {str(e)}")

    # 4. 문서 임베딩 생성
    try:
        doc_embeddings = sbert_model.encode(df["임베딩텍스트"].tolist(), convert_to_tensor=True)
    except Exception as e:
        raise RuntimeError(f"❌ 문서 임베딩 생성 실패: {str(e)}")

    # 5. 질문 임베딩 생성
    try:
        query_embedding = sbert_model.encode(query, convert_to_tensor=True)
    except Exception as e:
        raise RuntimeError(f"❌ 질문 임베딩 생성 실패: {str(e)}")

    # 6. 유사도 계산
    try:
        similarities = cosine_similarity(
            query_embedding.cpu().numpy().reshape(1, -1), 
            doc_embeddings.cpu().numpy()
        )[0]
    except Exception as e:
        raise RuntimeError(f"❌ 유사도 계산 중 오류: {str(e)}")

    # 7. 상위 top_k 인덱스 추출
    top_k_indices = np.argsort(similarities)[::-1][:top_k]

    # 8. 결과 DataFrame 반환
    try:
        top_docs = df.iloc[top_k_indices].copy()
        top_docs["유사도"] = similarities[top_k_indices]
    except Exception as e:
        raise RuntimeError(f"❌ 결과 데이터프레임 생성 실패: {str(e)}")

    if verbose:
        print("📄 Top 문서 목록:")
        print(top_docs[["파일명", "유사도"]])

    return top_docs

from src.utils.path import get_project_root_dir

def data_process(df: pd.DataFrame, apply_ocr: bool = True, file_type: str = "all") -> pd.DataFrame:
    """
    HWP 또는 PDF 파일을 처리하여 텍스트를 추출하고 full_text 컬럼에 저장합니다.

    Args:
        df (pd.DataFrame): 파일 목록을 포함한 데이터프레임
        apply_ocr (bool): PDF OCR 여부
        file_type (str): 'hwp', 'pdf', 'all' 중 하나

    Returns:
        pd.DataFrame: 텍스트가 추가된 DataFrame
    """
    base_dir = get_project_root_dir()
    file_root = os.path.join(base_dir, "data", "files")

    if file_type in ["hwp", "pdf"]:
        mask = df['파일명'].str.lower().str.endswith(f".{file_type}")
        filtered_df = df[mask].copy()
    elif file_type == "all":
        filtered_df = df.copy()

    filtered_df['full_text'] = None

    for file_name in filtered_df['파일명']:
        file_path = os.path.join(file_root, file_name)
        try:
            if not os.path.exists(file_path):
                 raise FileNotFoundError(f"❌ [FileNotFound] (data_loader.data_process.path) 파일이 존재하지 않습니다: {file_path}")

            if file_name.lower().endswith(".hwp") and file_type in ["hwp", "all"]:
                loader = HWPLoader(file_path)
                docs = loader.load()
                if docs and isinstance(docs[0].page_content, str):
                    filtered_df.loc[filtered_df['파일명'] == file_name, 'full_text'] = docs[0].page_content
                else:
                    print(f"⚠️ [Warning] (data_loader.data_process.hwp) HWP 파일 무시됨 (내용 없음): {file_name}")

            elif file_name.lower().endswith(".pdf") and file_type in ["pdf", "all"]:
                text = extract_text_from_pdf(Path(file_path), apply_ocr=apply_ocr)
                filtered_df.loc[filtered_df['파일명'] == file_name, 'full_text'] = text

            else:
                print(f"⚠️ [Warning] (data_loader.data_process) 지원하지 않는 파일 형식입니다: {file_name}")

        except Exception as e:
            raise RuntimeError(f"❌ [Runtime] (data_loader.data_process) 파일 처리 오류 ({file_name}): {e}")  

    return filtered_df.reset_index(drop=True)
