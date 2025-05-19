from typing import List
from langchain.schema import Document
from langchain.vectorstores.faiss import FAISS

def build_prompt_with_expansion(
    question: str,
    retrieved_docs: List[Document],
    all_docs: List[Document],
    window: int = 1,
    include_source: bool = True,
    prompt_template: str = None
) -> str:
    """
    검색된 문서 청크의 앞뒤 ±N 범위 인접 청크를 포함해 프롬프트를 구성합니다.

    Args:
        question (str): 사용자 질문
        retrieved_docs (List[Document]): 검색된 핵심 청크들
        all_docs (List[Document]): 전체 청크 리스트
        window (int): 인접 청크 포함 범위
        include_source (bool): 출처 포함 여부
        prompt_template (str): 사용자 정의 프롬프트 템플릿

    Returns:
        str: 구성된 프롬프트 문자열
    """
    used_chunks = set()
    context_blocks = []

    for doc in retrieved_docs:
        file_name = doc.metadata.get("파일명")
        base_idx = doc.metadata.get("chunk_idx", 0)

        if window == 0:
            neighbors = [doc]
        else:
            neighbors = [
                d for d in all_docs
                if d.metadata.get("파일명") == file_name and
                   abs(d.metadata.get("chunk_idx", -1) - base_idx) <= window
            ]

        for chunk in neighbors:
            key = (chunk.metadata.get("파일명"), chunk.metadata.get("chunk_idx"))
            if key in used_chunks:
                continue
            used_chunks.add(key)

            source_info = ""
            if include_source:
                source_info = (
                    f"[출처: {chunk.metadata.get('파일명')} | "
                    f"기관: {chunk.metadata.get('발주 기관')} | "
                    f"사업명: {chunk.metadata.get('사업명')}]"
                )

            context = f"{source_info}\n{chunk.page_content}".strip()
            context_blocks.append(context)

    context_text = "\n\n---\n\n".join(context_blocks)

    if prompt_template is None:
        prompt_template = (
            "아래는 정부 및 대학 관련 사업 문서 요약입니다. 문서 내용을 바탕으로 다음 질문에 대해 명확하고 핵심적으로 답변하세요.\n"
            "반복하지 말고, 문서에 포함된 정보만 기반으로 답하세요. 문서에 없는 정보는 '해당 문서에 정보가 없습니다.'라고 하세요.\n\n"
            "### 문서 내용:\n{context}\n\n"
            "### 질문:\n{question}\n\n"
            "### 답변 (문장으로 1~3줄):"
        )

    return prompt_template.format(context=context_text, question=question)


def get_all_documents_from_vectorstore(vectorstore: FAISS) -> List[Document]:
    """
    FAISS 벡터스토어에서 전체 Document 리스트를 반환합니다.

    Args:
        vectorstore (FAISS): 벡터스토어 객체

    Returns:
        List[Document]: 전체 문서 청크 리스트
    """
    if not hasattr(vectorstore, "docstore") or not hasattr(vectorstore.docstore, "_dict"):
        raise ValueError("해당 벡터스토어는 docstore 정보를 제공하지 않습니다.")

    return list(vectorstore.docstore._dict.values())