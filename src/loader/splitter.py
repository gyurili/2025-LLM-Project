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
    
    ####################################
    # 허용된 문자 패턴
    allowed_pattern = r"[^\uAC00-\uD7A3a-zA-Z0-9\s.,:;!?()~\-/]"

    # 제거 대상 문자 추출
    removed_chars = re.findall(allowed_pattern, text)
    if removed_chars:
        unique_removed = sorted(set(removed_chars))
        print(f"⚠️ 제거된 특수문자: {' '.join(unique_removed)}")

    # 1. 제거
    text = re.sub(allowed_pattern, " ", text)

    # 2. 연속된 공백 하나로 통일
    text = re.sub(r"\s+", " ", text)

    # 3. 앞뒤 공백 제거
    return text.strip()

# 1. 목차 기반 섹션 추출
def extract_sections(text: str) -> List[dict]:
    section_pattern = re.compile(r'\n?(\d+(\.\d+)*\s?[.)]?\s+[^\n]{2,50})')
    matches = list(section_pattern.finditer(text))

    chunks = []
    for i in range(len(matches)):
        start = matches[i].start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        title = matches[i].group().strip()
        content = text[start:end].strip()
        chunks.append({'title': title, 'content': content})
    return chunks

# 2. 너무 짧은 청크 병합
def merge_short_chunks(chunks: List[dict], min_length: int = 500) -> List[dict]:
    merged = []
    buffer = ""
    current_title = ""
    for chunk in chunks:
        if len(chunk["content"]) < min_length:
            buffer += " " + chunk["content"]
        else:
            if buffer:
                chunk["content"] = buffer.strip() + " " + chunk["content"]
                buffer = ""
            merged.append(chunk)
    return merged

# 3. 길이 초과 청크 재분할
def refine_chunks_with_length_control(chunks: List[dict], max_length: int = 1000, overlap: int = 250) -> List[dict]:
    refined = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=max_length, chunk_overlap=overlap)

    for chunk in chunks:
        split_texts = splitter.split_text(chunk["content"])
        for i, split_text in enumerate(split_texts):
            refined.append({
                "title": chunk["title"],
                "content": split_text,
                "sub_chunk_idx": i
            })
    return refined

# 4. 메인 함수
def data_chunking(df: pd.DataFrame, splitter_type: str = "section", size: int = 1000, overlap: int = 250) -> List[Document]:
    if splitter_type == "recursive":
        splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    elif splitter_type == "token":
        splitter = TokenTextSplitter(chunk_size=size, chunk_overlap=overlap)
    elif splitter_type == "section":
        splitter = None  # Custom 로직 사용
    else:
        raise ValueError(f"❌ [Value] (splitter.data_chunking.splitter_type) {splitter_type}은 지원하지 않습니다.")

    all_chunks = []
    for _, row in df.iterrows():
        text = row.get("full_text", "")
        if isinstance(text, str) and text.strip():
            try:
                text = clean_text(text)  # 사전 정의된 전처리 함수
                if splitter_type == "section":
                    sections = extract_sections(text)
                    merged = merge_short_chunks(sections)
                    chunks = refine_chunks_with_length_control(merged, max_length=size, overlap=overlap)
                else:
                    chunks = splitter.split_text(text)

                for i, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk["content"] if isinstance(chunk, dict) else chunk,
                        metadata={
                            "사업명": row.get("사업명", ""),
                            "발주 기관": row.get("발주 기관", ""),
                            "파일명": row.get("파일명", ""),
                            "chunk_idx": i,
                            "chunk_title": chunk.get("title", "") if isinstance(chunk, dict) else "",
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
        "2번째 청크": file_chunks[1] if len(file_chunks) > 1 else None,
        "3번째 청크": file_chunks[2] if len(file_chunks) > 2 else None,
        "4번째 청크": file_chunks[3] if len(file_chunks) > 3 else None,
        "5번째 청크": file_chunks[4] if len(file_chunks) > 4 else None,
        "중간 청크": file_chunks[len(file_chunks) // 2],
        "마지막 청크": file_chunks[-1],
        "가장 긴 청크": file_chunks[idx_max],
        "가장 짧은 청크": file_chunks[idx_min],
    }
    
    for label, doc in selected.items():
        print(f"        - {label} 길이: {len(doc.page_content)}")
        print(f"        - 내용: {doc.page_content[:1000] + ('...' if len(doc.page_content) > 1000 else '')}")


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