import os
import pandas as pd
from typing import List
from dotenv import load_dotenv
from pathlib import Path
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
    주어진 경로에서 CSV 파일을 불러오고, 필요한 컬럼만 추출하여 전처리된 DataFrame을 반환합니다.

    Args:
        path (str): 불러올 CSV 파일의 상대 경로

    Returns:
        pd.DataFrame: 전처리된 데이터프레임

    Raises:
        FileNotFoundError: 파일이 존재하지 않을 경우
        pd.errors.ParserError: CSV 파싱 오류 발생 시
    """
    if '__file__' in globals():
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    else:
        base_dir = os.path.abspath("..")

    full_path = os.path.join(base_dir, path)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {full_path}")

    df = pd.read_csv(full_path)

    required_columns = ['파일명', '사업 요약', '텍스트', '사업명', '발주 기관', '사업 금액']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"필수 컬럼이 누락되었습니다: {set(required_columns) - set(df.columns)}")

    df = df[required_columns]
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
    if '__file__' in globals():
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    else:
        base_dir = os.path.abspath("..")

    file_root = os.path.join(base_dir, "data", "files")

    for file_name in df['파일명']:
        file_path = os.path.join(file_root, file_name)

        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"파일이 존재하지 않습니다: {file_path}")

            if file_name.lower().endswith(".hwp"):
                loader = HWPLoader(file_path)
                docs = loader.load()
                if docs and isinstance(docs[0].page_content, str):
                    df.loc[df['파일명'] == file_name, 'full_text'] = docs[0].page_content
                else:
                    print(f"HWP 파일 무시됨 (내용 없음): {file_name}")

            elif file_name.lower().endswith(".pdf"):
                text = extract_text_from_pdf(Path(file_path), apply_ocr=apply_ocr)
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