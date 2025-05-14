import os
from typing import List

import pandas as pd
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_teddynote.document_loaders import HWPLoader
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
import faiss
from pdf_loader import extract_text_from_pdf

def data_load(path: str) -> pd.DataFrame:
    """
    CSV 파일을 불러와 필요한 컬럼을 포함한 전처리된 DataFrame을 반환합니다.

    Args:
        path (str): 프로젝트 루트 기준 CSV 파일 상대 경로 (예: "data/data_list.csv")

    Returns:
        pd.DataFrame: 전처리된 데이터프레임

    Raises:
        FileNotFoundError: 파일이 존재하지 않을 경우
        pd.errors.ParserError: CSV 파일 파싱 중 오류가 발생한 경우
        ValueError: 필수 컬럼이 누락된 경우
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_path = os.path.join(base_dir, path)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"CSV 파일이 존재하지 않습니다: {full_path}")

    try:
        df = pd.read_csv(full_path)
    except pd.errors.ParserError as e:
        raise pd.errors.ParserError(f"CSV 파싱 오류: {e}")

    required_columns = ['파일명', '사업 요약', '텍스트', '사업명', '발주 기관', '사업 금액']
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"필수 컬럼이 누락되었습니다: {missing_cols}")

    df = df[required_columns].copy()
    df['사업 금액'] = pd.to_numeric(df['사업 금액'], errors='coerce').astype("Int64")
    return df


def data_process(df: pd.DataFrame, apply_ocr: bool = True) -> pd.DataFrame:
    """
    HWP 또는 PDF 파일을 처리하여 텍스트를 추출하고,
    'full_text' 컬럼에 저장합니다.

    Args:
        df (pd.DataFrame): 파일명 컬럼이 포함된 DataFrame
        apply_ocr (bool): PDF 처리 시 OCR을 적용할지 여부

    Returns:
        pd.DataFrame: 텍스트가 추가된 DataFrame
    """
    base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/files")

    df["full_text"] = ""

    for file_name in df["파일명"]:
        file_path = os.path.join(base_path, file_name)

        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"파일이 존재하지 않습니다: {path}")

            if file_name.lower().endswith(".hwp"):
                loader = HWPLoader(path)
                docs = loader.load()
                if docs and isinstance(docs[0].page_content, str):
                    df.loc[df['파일명'] == file_name, 'full_text'] = docs[0].page_content
                else:
                    print(f"HWP 파일 무시됨 (내용 없음): {file_name}")

            elif file_name.lower().endswith(".pdf"):
                text = extract_text_from_pdf(Path(path), apply_ocr=apply_ocr)
                df.loc[df['파일명'] == file_name, 'full_text'] = text

            else:
                print(f"지원하지 않는 파일 형식: {file_name}")

        except Exception as e:
            print(f"파일 처리 오류 ({file_name}): {e}")

    return df


def data_chunking(df: pd.DataFrame) -> List[Document]:
    """
    full_text 컬럼을 기준으로 텍스트를 청크로 분할하고 Document 객체로 반환합니다.

    Args:
        df (pd.DataFrame): 'full_text'가 포함된 DataFrame

    Returns:
        List[Document]: 청크 단위로 나뉜 Document 객체 리스트
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    all_chunks = []

    for _, row in df.iterrows():
        text = row.get("full_text", "")
        if isinstance(text, str) and text.strip():
            try:
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
                print(f"[청크 처리 오류] {row.get('파일명')}: {e}")
        else:
            print(f"[스킵됨] 텍스트 없음: {row.get('파일명')}")
    return all_chunks