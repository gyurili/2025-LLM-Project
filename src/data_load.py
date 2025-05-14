import os
from typing import List

import pandas as pd
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_teddynote.document_loaders import HWPLoader


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

def data_process(df: pd.DataFrame) -> pd.DataFrame:
    """
    DataFrame 내 파일명을 기반으로 HWP 파일을 로드하여 'full_text' 컬럼에 텍스트 추가

    Args:
        df (pd.DataFrame): 파일명 정보를 포함한 DataFrame

    Returns:
        pd.DataFrame: 각 행에 'full_text' 컬럼이 추가된 DataFrame
    """
    base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/files")

    df["full_text"] = ""

    for file_name in df["파일명"]:
        file_path = os.path.join(base_path, file_name)

        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"HWP 파일이 존재하지 않습니다: {file_path}")

            loader = HWPLoader(file_path)
            docs = loader.load()

            content = docs[0].page_content if docs and isinstance(docs[0].page_content, str) else ""
            df.loc[df["파일명"] == file_name, "full_text"] = content

            if not content.strip():
                print(f"내용 없음 - 무시됨: {file_name}")

        except Exception as e:
            print(f"[오류] {file_name}: {e}")

    return df

def hwp_chunking(df: pd.DataFrame) -> List[Document]:
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