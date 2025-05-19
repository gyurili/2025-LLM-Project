import re
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter

'''
    To do:
        1. verbose일때 청킹이 잘 되었는지 확인하는 로깅 추가
        2. 의미 단위로 청크를 나누는 방법 추가
'''

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
        raise ValueError("❌ [Type] (splitter.clean_text) 입력값은 문자열이어야 합니다.")
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
        raise ValueError(f"❌ [Value] (splitter.data_chunking.splitter_type) {splitter_type}은 지원하지 않는 청크 분할기입니다.")

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
                raise RuntimeError(f"❌ [Runtime] (splitter.data_chunking) 청크 생성 오류 ({row.get('파일명')}): {e}")
        else:
            raise ValueError(f"❌ [Data] (splitter.data_chunking) full_text가 비어있거나 문자열이 아닙니다: {row.get('파일명')}")
        
    return all_chunks


def insepect_sample_chunks(chunks: List[Document], file_name: str) -> None:
    """
    특정 파일의 청크를 샘플로 출력합니다.

    Args:
        chunks (List[Document]): Document 객체 리스트
        file_name (str): 검사할 파일명
    """
    file_chunks = [doc for doc in chunks if doc.metadata.get("파일명") == file_name]
    if not file_chunks:
        print(f"❌ [Data] (splitter.insepect_sample_chunks) {file_name}에 대한 청크가 없습니다.")
        return
    
    lengths = [len(doc.page_content) for doc in file_chunks]
    idx_max = lengths.index(max(lengths))
    idx_min = lengths.index(min(lengths))

    selected = {
        "첫 청크": file_chunks[0],
        "중간 청크": file_chunks[len(file_chunks) // 2],
        "마지막 청크": file_chunks[-1],
        "가장 긴 청크": file_chunks[idx_max],
        "가장 짧은 청크": file_chunks[idx_min],
    }
    
    for label, doc in selected.items():
        print(f"        - {label} 길이: {len(doc.page_content)}")
        print(f"        - 내용: {doc.page_content[:300] + ('...' if len(doc.page_content) > 300 else '')}")


def summarize_chunk_quality(chunks: List[Document], verbose: bool = False):
    """
    청크의 품질을 요약하여 콘솔에 출력합니다.

    Args:
        chunks (List[Document]): Document 객체 리스트
        verbose (bool): 샘플 출력 여부
    """
    summary = defaultdict(list)
    
    for doc in chunks:
        file_name = doc.metadata.get("파일명", "Unknown")
        length = len(doc.page_content)
        summary[file_name].append(length)

    # 평균 길이 계산
    results = []
    for fname, lengths in summary.items():
        arr = np.array(lengths)
        results.append({
            "파일명": fname,
            "청크수": len(arr),
            "평균길이": np.mean(arr),
            "최소길이": np.min(arr),
            "최대길이": np.max(arr),
            "500자미만비율": np.sum(arr < 500) / len(arr) * 100,
        })

    results.sort(key=lambda x: x["500자미만비율"], reverse=True)

    print("    - 청크 품질 요약:")
    for res in results:
        print(f"    - {res['파일명']}")
        print(f"        - 청크수: {res['청크수']}")
        print(f"        - 평균길이: {res['평균길이']}")
        print(f"        - 최소길이: {res['최소길이']}")
        print(f"        - 최대길이: {res['최대길이']}")
        print(f"        - 500자미만비율: {res['500자미만비율']:.2f}%")

        if verbose:
            insepect_sample_chunks(chunks, res['파일명'])
            print("-" * 30)