import os
import fitz
import faiss
import easyocr
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_teddynote.document_loaders import HWPLoader
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

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


def data_load(path: str, limit: int = None) -> pd.DataFrame:
    """
    주어진 경로에서 CSV 파일을 불러와 전처리합니다.

    Args:
        path (str): CSV 파일 상대 경로
        limit (Optional[int]): 데이터프레임의 행 수 제한 (기본값: None)

    Returns:
        pd.DataFrame: 전처리된 데이터프레임
    """
    if '__file__' in globals():
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    else:
        base_dir = os.path.abspath("..")
    full_path = os.path.join(base_dir, path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"❌ [FileNotFound] (data_loader.data_load.path) CSV 파일을 찾을 수 없습니다: {full_path}")
    if limit < 1:
        limit = 1
        print("⚠️ [Warning] (data_loader.data_load.limit) limit은 0보다 큰 정수여야 합니다. 최소값 1로 설정합니다.")
    elif limit > 100:
        limit = 100
        print("⚠️ [Warning] (data_loader.data_load.limit) limit은 100보다 작거나 같아야 합니다. 최대값 100으로 설정합니다.")

    df = pd.read_csv(full_path)
    required_columns = ['파일명', '사업 요약', '텍스트', '사업명', '발주 기관', '사업 금액']
    if not all(col in df.columns for col in required_columns):
        missing = set(required_columns) - set(df.columns)
        raise ValueError(f"❌ [Value] (data_loader.data_load.columns) 필수 컬럼이 누락되었습니다: {missing}") 

    df = df[required_columns]
    df['사업 금액'] = pd.to_numeric(df['사업 금액'], errors='coerce').astype("Int64")
    if limit is not None:
        df = df.head(limit)
    return df


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
    if '__file__' in globals():
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    else:
        base_dir = os.path.abspath("..")
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


import re

def clean_text(text: str) -> str:
    """
    입력 문자열에서 불필요한 문자 및 공백을 정리합니다.

    처리 단계:
    1. 한자 및 유니코드 특수문자를 제거하고, 
       한글, 영문자, 숫자, 공백 및 일부 특수문자(.,:;!?()~-/)만 남깁니다.
    2. 연속된 공백을 하나의 공백으로 통일합니다.
    3. 문자열 양 끝의 공백을 제거합니다.

    Args:
        text (str): 전처리할 원본 문자열

    Returns:
        str: 정제된 문자열
    """
    if not isinstance(text, str):
        raise ValueError("❌ [Type] (data_loader.clean_text) 입력값은 문자열이어야 합니다.")
    # 1. 한자 및 유니코드 특수문자 제거 (한글, 영어, 숫자, 공백, 일부 특수문자 제외)
    text = re.sub(r"[^\uAC00-\uD7A3a-zA-Z0-9\s.,:;!?()~\-/]", " ", text)

    # 2. 연속된 공백 하나로 통일
    text = re.sub(r"\s+", " ", text)

    # 3. 앞뒤 공백 제거
    return text.strip()


def data_chunking(df: pd.DataFrame, splitter_type: str = "recursive", size: int = 300, overlap: int = 50) -> List[Document]:
    """
    full_text 컬럼을 기준으로 텍스트를 청크로 분할하고 Document 객체로 반환합니다.

    Args:
        df (pd.DataFrame): 텍스트가 포함된 DataFrame

    Returns:
        List[Document]: 청크 단위로 나뉜 Document 객체 리스트
    """
    if splitter_type == "recursive":
        splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    elif splitter_type == "token":
        splitter = TokenTextSplitter(chunk_size=size, chunk_overlap=overlap)
    else:
        raise ValueError(f"❌ [Value] (data_loader.data_chunking.splitter_type) {splitter_type}은 지원하지 않는 청크 분할기입니다.")

    all_chunks = []
    for _, row in df.iterrows():
        text = row.get("full_text", "")
        if isinstance(text, str) and text.strip():
            try:
                text = clean_text(text)
                chunks = splitter.split_text(text)
                for i, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "사업명": row.get("사업명", ""),
                            "발주 기관": row.get("발주 기관", ""),
                            "파일명": row.get("파일명", ""),
                            "chunk_idx": i
                        }
                    )
                    all_chunks.append(doc)
            except Exception as e:
                raise RuntimeError(f"❌ [Runtime] (data_loader.data_chunking) 청크 생성 오류 ({row.get('파일명')}): {e}")
        else:
            raise ValueError(f"❌ [Data] (data_loader.data_chunking) full_text가 비어있거나 문자열이 아닙니다: {row.get('파일명')}")
    return all_chunks