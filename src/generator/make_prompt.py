from typing import List
from langchain.schema import Document


'''
    To Do:
        - 입력된 질문에 따라 프롬프트를 구성하는 기능
'''

def build_prompt(
    question: str,
    retrieved_docs: List[Document],
    include_source: bool = True,
    prompt_template: str = None,
    chat_history: str = None,
) -> str:
    """
    주어진 질문과 검색된 문서들을 기반으로 프롬프트를 구성합니다.
    
    Args:
        question (str): 사용자 질문
        retrieved_docs (List[Document]): 검색된 핵심 청크들
        include_source (bool): 출처 포함 여부
        prompt_template (str): 사용자 정의 프롬프트 템플릿

    Returns:
        str: 구성된 프롬프트 문자열
    """
    if not retrieved_docs:
        raise ValueError("❌ (generator.make_prompt.build_prompt) 검색된 문서가 없습니다.")

    context_blocks = []

    for chunk in retrieved_docs:
        source_info = ""
        if include_source:
            source_info = (
                f"[출처: {chunk.metadata.get('파일명')} | "
                f"기관: {chunk.metadata.get('발주 기관')} | "
                f"사업명: {chunk.metadata.get('사업명')} | "
                f"청크 번호: {chunk.metadata.get('chunk_idx')}]"
            )

        context = f"{source_info}\n{chunk.page_content}".strip()
        context_blocks.append(context)

    context_text = "\n\n---\n\n".join(context_blocks)

    # 대화 내역 요약 검사
    chat_history_section = ""
    if chat_history:
        chat_history_section = f"{chat_history}"

    if prompt_template is None:
        prompt_template = (
            "당신은 정부 및 대학의 공공 사업 제안서를 분석하는 AI 전문가입니다.\n"
            "아래 문서 내용은 특정 사업의 목적, 예산, 수행 방식 등을 요약한 것입니다.\n"
            "※ 아래 지침을 철저히 따르세요. 위반 시 틀린 답변으로 간주됩니다.\n\n"
            "다음 질문에 대해 다음 원칙에 따라 명확하고 정확하게 답변하세요:\n"
            "- 반드시 문서 내용에 기반해서만 답하세요.\n"
            "- 문서를 참고했다면 출처를 표기하세요.\n"
            "- 문서에 정보가 없으면 '해당 문서에 정보가 없습니다.'라고 말하세요.\n"
            "- 불확실하거나 추측되는 내용은 포함하지 마세요.\n"
            "- 문서 외의 지식, 상식, 다른 문서나 유사 사례를 근거로 답하지 마세요\n"
            "- 답변은 최대 5문장 이내로 작성하세요.\n"
            "- 항목이 여러 개인 경우, 항목별로 줄바꿈하여 나열하세요.\n"
            "- 답변의 마지막엔 출처가 된 문서들을 [출처: '문서명', 청크번호: '숫자'] 형식으로 작성 하시오.\n"
            "- 문서 내용 중간에 출처를 표시하지 마세요.\n\n"
            " 이전에 사용자와 나눈 대화 내역이 아래에 정리되어 있습니다. 이 내역은 질문의 의도를 파악하는 데 도움이 됩니다. 그러나 여전히 답변은 반드시 문서 내용을 기반으로 해야 하며, 문서에 없는 내용은 포함하지 마십시오.\n\n"
            "### 이전 대화:\n{chat_history_section}\n\n"
            "### 문서 내용:\n{context}\n\n"
            "### 질문:\n{question}\n\n"
            "### 답변:"
        )

    return prompt_template.format(context=context_text, question=question, chat_history_section=chat_history_section)