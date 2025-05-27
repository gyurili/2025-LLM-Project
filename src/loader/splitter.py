import re
import numpy as np
import pandas as pd

from collections import defaultdict
from typing import List
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter


def clean_text(text: str) -> str:
    """
    입력 문자열에서 불필요한 문자 및 공백을 정리합니다.

    Args:
        text (str): 전처리할 원본 문자열

    Returns:
        str: 정제된 문자열
        
    Raises:
        ValueError: 입력이 문자열이 아닌 경우
    """
    if not isinstance(text, str):
        raise ValueError("❌ [Type] (splitter.clean_text) 문자열이 아닌 입력값")

    allowed_pattern = r"[^\uAC00-\uD7A3a-zA-Z0-9\s.,:;!?()\[\]~\-/•※❍□ㅇ○①-⑳IVXLCDM]"
    text = re.sub(allowed_pattern, " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_sections(text: str) -> List[dict]:
    """
    다양한 목차 패턴으로 텍스트를 분리합니다.

    Args:
        text (str): 전체 문서 텍스트

    Returns:
        List[dict]: 분리된 섹션 정보 리스트 (각 항목은 'title'과 'content' 포함)
    """
    section_pattern = re.compile(
        r"""
        ^[ \t]*
        (
            (?:\d+(?:\.\d+)*[.)]?) |
            (?:[가-힣]{1}[.)]?) |
            (?:\[\d+\]) |
            (?:\[\s*붙임\s*\d+\s*\]) |
            (?:[①-⑳]) |
            (?:[○•※❍□ㅇ]) |
            (?:[IVXLCDM]{1,7}\.?)
        )
        [ \t]+
        ([^\n]{2,50})
        """,
        re.MULTILINE | re.VERBOSE
    )

    matches = list(section_pattern.finditer(text))

    chunks = []
    for i in range(len(matches)):
        start = matches[i].start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        title = matches[i].group().strip()
        content = text[start:end].strip()
        chunks.append({"title": title, "content": content})
    return chunks


def merge_short_chunks(chunks: List[dict], min_length: int = 500) -> List[dict]:
    """
    길이가 min_length 미만인 청크들을 인접한 청크에 병합하여 반환합니다.

    Args:
        chunks (List[dict]): 섹션 단위로 분리된 청크 리스트
        min_length (int, optional): 병합 기준이 되는 최소 길이 (기본값: 500)

    Returns:
        List[dict]: 병합된 청크 리스트
    """
    merged = []
    buffer = ""
    for chunk in chunks:
        if len(chunk["content"]) < min_length:
            buffer += " " + chunk["content"]
        else:
            if buffer:
                chunk["content"] = buffer.strip() + " " + chunk["content"]
                buffer = ""
            merged.append(chunk)
    return merged


def refine_chunks_with_length_control(
    chunks: List[dict],
    max_length: int = 1000,
    overlap: int = 250
) -> List[dict]:
    """
    각 청크의 길이를 제한하면서 겹치는 영역을 포함해 추가 분할합니다.

    Args:
        chunks (List[dict]): 병합된 청크 리스트
        max_length (int, optional): 최대 청크 길이 (기본값: 1000)
        overlap (int, optional): 청크 간 중첩 길이 (기본값: 250)

    Returns:
        List[dict]: 길이 제한 및 중첩 처리가 적용된 청크 리스트
    """
    refined = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_length, chunk_overlap=overlap
    )

    for chunk in chunks:
        split_texts = splitter.split_text(chunk["content"])
        for i, split_text in enumerate(split_texts):
            refined.append(
                {
                    "title": chunk["title"],
                    "content": split_text,
                    "sub_chunk_idx": i,
                }
            )
    return refined


def data_chunking(
    df: pd.DataFrame,
    splitter_type: str = "section",
    size: int = 1000,
    overlap: int = 250,
) -> List[Document]:
    """
    데이터프레임의 각 row를 청크 단위로 분할하고, langchain Document로 변환합니다.

    Args:
        df (pd.DataFrame): 'full_text' 컬럼을 포함한 데이터프레임
        splitter_type (str, optional): 분할 방식 ('recursive', 'token', 'section')
        size (int, optional): 청크 최대 크기 (기본값: 1000)
        overlap (int, optional): 청크 간 중첩 길이 (기본값: 250)

    Returns:
        List[Document]: 청크 단위로 변환된 langchain Document 리스트

    Raises:
        ValueError: 텍스트가 비어있거나 splitter_type이 지원되지 않는 경우
        RuntimeError: 청크 생성 중 예외 발생 시
    """
    if splitter_type == "recursive":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size, chunk_overlap=overlap
        )
    elif splitter_type == "token":
        splitter = TokenTextSplitter(chunk_size=size, chunk_overlap=overlap)
    elif splitter_type == "section":
        splitter = None
    else:
        raise ValueError(
            f"❌ [Value] (splitter.data_chunking.splitter_type) 지원하지 않는 분할 방식: {splitter_type}"
        )

    all_chunks = []
    for _, row in df.iterrows():
        text = row.get("full_text", "")
        if isinstance(text, str) and text.strip():
            try:
                text = clean_text(text)
                if splitter_type == "section":
                    sections = extract_sections(text)
                    if not sections:
                        print(f"⚠️ [Skip] 섹션 추출 실패로 청크 없음: {row.get('파일명')}")    
                    merged = merge_short_chunks(sections)
                    chunks = refine_chunks_with_length_control(
                        merged, max_length=size, overlap=overlap
                    )
                    if not chunks:
                        print(f"⚠️ [Skip] 청크 0개 생성됨: {row.get('파일명')}")
                else:
                    chunks = splitter.split_text(text)

                for i, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk["content"]
                        if isinstance(chunk, dict)
                        else chunk,
                        metadata={
                            "사업명": row.get("사업명", ""),
                            "발주 기관": row.get("발주 기관", ""),
                            "파일명": row.get("파일명", ""),
                            "chunk_idx": i,
                            "chunk_title": chunk.get("title", "")
                            if isinstance(chunk, dict)
                            else "",
                        },
                    )
                    all_chunks.append(doc)
            except Exception as e:
                raise RuntimeError(
                    f"❌ [Runtime] (splitter.data_chunking) 청크 생성 오류 ({row.get('파일명')}): {e}"
                )
        else:
            print(f"⚠️ [Skip] full_text 비어있어 청크 건너뜸: {row.get('파일명')}")
            continue  # 건너뛰기

    return all_chunks


def inspect_sample_chunks(
    chunks: List[Document], file_name: str, verbose: bool = False
) -> None:
    """
    특정 파일에 해당하는 청크들 중 주요 청크(첫, 중간, 마지막, 최대/최소 길이)를 출력합니다.

    Args:
        chunks (List[Document]): 전체 Document 청크 리스트
        file_name (str): 확인할 대상 파일명
        verbose (bool, optional): 출력 여부

    Returns:
        None
    """
    if not verbose:
        return

    file_chunks = [doc for doc in chunks if doc.metadata.get("파일명") == file_name]
    if not file_chunks:
        print(f"❌ [Data] (splitter.inspect_sample_chunks) 청크 없음: {file_name}")
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
        print(f"\n▶ {label}")
        print(f"  - 길이: {len(doc.page_content)}")
        print("  - 내용:")
        preview = doc.page_content[:300]
        if len(doc.page_content) > 300:
            preview += "..."
        print(f"    {preview}")


def summarize_chunk_quality(
    chunks: List[Document], verbose: bool = False
) -> None:
    """
    파일별로 청크 수, 평균/최소/최대 길이, 500자 미만 비율 등의 청크 품질 통계를 요약 출력합니다.

    Args:
        chunks (List[Document]): 전체 Document 청크 리스트
        verbose (bool, optional): 샘플 청크 출력 여부

    Returns:
        None
    """
    if not verbose:
        return

    summary = defaultdict(list)
    for doc in chunks:
        file_name = doc.metadata.get("파일명", "Unknown")
        length = len(doc.page_content)
        summary[file_name].append(length)

    results = []
    for fname, lengths in summary.items():
        arr = np.array(lengths)
        results.append(
            {
                "파일명": fname,
                "청크수": len(arr),
                "평균길이": np.mean(arr),
                "최소길이": np.min(arr),
                "최대길이": np.max(arr),
                "500자미만비율": np.sum(arr < 500) / len(arr) * 100,
            }
        )

    results.sort(key=lambda x: x["500자미만비율"], reverse=True)

    print("\n📌 청크 품질 요약")
    for res in results:
        print("=" * 60)
        print(f"📄 파일명: {res['파일명']}")
        print(f"  - 청크 수         : {res['청크수']}")
        print(f"  - 평균 길이       : {res['평균길이']:.2f}")
        print(f"  - 최소 길이       : {res['최소길이']}")
        print(f"  - 최대 길이       : {res['최대길이']}")
        print(f"  - 500자 미만 비율 : {res['500자미만비율']:.2f}%")
        inspect_sample_chunks(chunks, res['파일명'], verbose=True)