import re
from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd
from langchain.schema import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)



def clean_text(text: str) -> str:
    """
    입력 문자열에서 불필요한 문자 및 공백을 정리합니다.

    Args:
        text (str): 전처리할 원본 문자열

    Returns:
        str: 정제된 문자열
    """
    if not isinstance(text, str):
        raise ValueError("❌ [Type] (splitter.clean_text) 입력값은 문자열이어야 합니다.")

    allowed_pattern = r"[^\uAC00-\uD7A3a-zA-Z0-9\s.,:;!?()~\-/]"
    text = re.sub(allowed_pattern, " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()



def extract_sections(text: str) -> List[dict]:
    section_pattern = re.compile(r"\n?(\d+(\.\d+)*\s?[.)]?\s+[^\n]{2,50})")
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
    chunks: List[dict], max_length: int = 1000, overlap: int = 250
) -> List[dict]:
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
            f"❌ [Value] (splitter.data_chunking.splitter_type) {splitter_type}은 지원하지 않습니다."
        )

    all_chunks = []
    for _, row in df.iterrows():
        text = row.get("full_text", "")
        if isinstance(text, str) and text.strip():
            try:
                text = clean_text(text)
                if splitter_type == "section":
                    sections = extract_sections(text)
                    merged = merge_short_chunks(sections)
                    chunks = refine_chunks_with_length_control(
                        merged, max_length=size, overlap=overlap
                    )
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
            raise ValueError(
                f"❌ [Data] (splitter.data_chunking) full_text가 비어있거나 문자열이 아닙니다: {row.get('파일명')}"
            )

    return all_chunks



def inspect_sample_chunks(
    chunks: List[Document], file_name: str, verbose: bool = False
) -> None:
    if not verbose:
        return

    file_chunks = [doc for doc in chunks if doc.metadata.get("파일명") == file_name]
    if not file_chunks:
        print(
            f"❌ [Data] (splitter.inspect_sample_chunks) {file_name}에 대한 청크가 없습니다."
        )
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
        preview = doc.page_content[:500]
        if len(doc.page_content) > 500:
            preview += "..."
        print(f"        - 내용: {preview}")



def summarize_chunk_quality(
    chunks: List[Document], verbose: bool = False
) -> None:
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

    print("    - 청크 품질 요약:")
    for res in results:
        print(f"    - {res['파일명']}")
        print(f"        - 청크수: {res['청크수']}")
        print(f"        - 평균길이: {res['평균길이']}")
        print(f"        - 최소길이: {res['최소길이']}")
        print(f"        - 최대길이: {res['최대길이']}")
        print(f"        - 500자미만비율: {res['500자미만비율']:.2f}%")
        inspect_sample_chunks(chunks, res['파일명'], verbose=True)
        print("-" * 30)