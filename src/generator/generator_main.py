from typing import List
from collections import defaultdict
from langchain.schema import Document
from src.generator.hf_generator import load_hf_model, generate_answer_hf
from src.generator.openai_generator import load_openai_model, generate_answer_openai
from src.generator.make_prompt import build_prompt


def summarize_chat_history(config, only_user=False):
    """
    Langchain Document 또는 단순 문자열 기반 요약 생성
    """
    from src.generator.hf_generator import generate_answer_hf
    from src.generator.openai_generator import generate_answer_openai

    history_text = "\n".join(
        [f"{'질문' if turn['role'] == 'user' else '답변'}: {turn['content']}" for turn in config["chat_history"]]
    )
    if only_user:
        prompt = f"다음은 사용자의 질문 내용입니다. 이 질문의 핵심 내용을 간결하게 요약해 주세요.\n\n{history_text}"
    else:
        prompt = f"다음은 사용자와 AI의 대화 내용입니다. 이 대화의 핵심 내용을 간결하게 요약해 주세요.\n\n{history_text}"

    if config["generator"]["model_type"] == "huggingface":
        model_info = load_hf_model(config)
        return generate_answer_hf(prompt, model_info, config["generator"])
    elif config["generator"]["model_type"] == "openai":
        model_info = load_openai_model(config)
        return generate_answer_openai(prompt, model_info, config["generator"])


def load_chat_history(config):
    """
    과거 질문, 답변 내역을 로드합니다.
    """
    # 1. 과거 질문, 답변 내역
    if config["chat_history"]:
        chat_history_str = "\n".join([f"질문: {turn['content']}" for turn in config["chat_history"]])

        # 2. 내역 요약
        chat_history_str = summarize_chat_history(config)
        print(f"과거 대화 내역 요약: {chat_history_str}")
        return chat_history_str
    else: # 대화 내역이 없을 경우
        chat_history_str = ""
        return chat_history_str
    

def load_user_query(config):
    """
    사용자 질문을 로드합니다.
    """
    # 1. 사용자 질문
    if config["chat_history"]:
        user_query = "\n".join([f"질문: {turn['content']}" for turn in config["chat_history"] if turn["role"] == "user"])

        # 2. 내역 요약
        user_query = summarize_chat_history(config, only_user=True)
        return user_query
    else: # 대화 내역이 없을 경우
        user_query = ""
        return user_query


def generator_main(
    retrieved_docs: List[Document],
    config: dict,
    model_info=None
) -> str:
    """
    검색된 문서 리스트를 기반으로 답변을 생성하는 메인 실행 함수.

    Args:
        retrieved_docs (List[Document]): 검색된 핵심 청크 리스트
        all_docs (List[Document]): 전체 문서 청크 리스트
        config (dict): 설정 정보 (모델, 템플릿 등 포함)

    Returns:
        str: 생성된 답변
    """
    # 0. 과거 질의응답 내역 요약
    chat_history = load_chat_history(config)
    print(f"과거 맥락 :{chat_history}")

    # 1. 프롬프트 생성
    query = config["retriever"]["query"]

    prompt = build_prompt(
        question=query,
        retrieved_docs=retrieved_docs,
        include_source=config.get("include_source", True),
        prompt_template=config.get("prompt_template"),
        chat_history=chat_history,
    )

    model_type = config["generator"]["model_type"]
    if model_info is None:
        if model_type == "huggingface":
            model_info = load_hf_model(config)
        elif model_type == "openai":
            model_info = load_openai_model(config)
            
    if model_type == "huggingface":
        answer = generate_answer_hf(prompt, model_info, config["generator"])
    elif model_type == "openai":
        answer = generate_answer_openai(prompt, model_info, config["generator"])
        
    print(answer)
    return answer


def is_unsatisfactory(answer: str) -> bool:
    if '정보가 없습니다' in answer or '잘 모르겠습니다' in answer:
        print("Misunderstanding Dectection")
        return True
    if len(answer.strip()) < 10:
        print("Short Answer Dectection")
        return True
    
    # 반복 문장 감지
    lines = answer.strip().splitlines()
    unique_lines = set(line.strip() for line in lines if line.strip())
    if len(unique_lines) < len(lines) / 2:
        return True

    return False


def generate_with_clarification(
    retrieved_docs: List[Document], 
    config: dict,
    max_retries: int = 1,
    model_info = None
) -> str:
    """
    검색된 문서와 초기 질문을 바탕으로 생성된 답변이 부적절할 경우,
    issue를 정의하고 clarification 프롬프트를 사용하여 재생성 시도.

    Args:
        retrieved_docs (List[Document]): 검색된 문서들
        config (dict): 설정 정보 (retriever/generator 관련 포함)
        max_retries (int): 최대 재생성 시도 횟수
        model_info: 미리 로드된 생성 모델 정보 (옵션)

    Returns:
        str: 최종 생성된 적절한 답변
    """
    # 0) 대화 내역 요약
    chat_history = load_chat_history(config)

    # 1) context 구성
    docs_by_file = defaultdict(list)
    for chunk in retrieved_docs:
        file_name = chunk.metadata.get("파일명", "알 수 없음")
        docs_by_file[file_name].append(chunk.page_content)

    context_blocks = []
    for file_name, chunks in docs_by_file.items():
        source_info = f"[출처: {file_name}]"
        document_text = "\n".join(chunks)
        context_blocks.append(f"{source_info}\n{document_text}")

    context_text = "\n\n---\n\n".join(context_blocks)

    clarification_template = (
        "이전 답변이 적절하지 않았습니다. 다음 이슈를 고려하여 더 명확하고 근거 있는 답변을 생성해주세요:\n"
        "{issue}\n\n"
        "- 아래 각 context 블록은 서로 다른 문서에서 추출된 내용입니다.\n"
        "- 반드시 문서에 기반한 내용만 답변에 포함하고, 출처를 바탕으로 신뢰도 있게 작성하세요.\n"
        "- 사전 지식이나 추측은 절대 포함하지 마세요.\n"
        "- 답변은 한국어로 작성하세요.\n\n"
        "### 이전 대화:\n{chat_history}\n\n"
        "### 문서 내용:\n{context}\n\n"
        "### 질문:\n{question}\n\n"
        "### 보완된 답변:"
    )

    # 2) 초기 답변 생성 (model_info 없으면 최초 로드)
    model_type = config["generator"]["model_type"]
    if model_info is None:
        if model_type == "huggingface":
            model_info = load_hf_model(config)
        elif model_type == "openai":
            model_info = load_openai_model(config)
        else:
            raise ValueError(f"지원되지 않는 generator model_type: {model_type}")

    answer = generator_main(retrieved_docs, config, model_info)

    # 3) 재시도 루프
    for i in range(max_retries):
        if not is_unsatisfactory(answer):
            break  # 만족하는 답변이면 종료

        print(f"⚠️ 답변 재생성 중...({i+1}/{max_retries})")
        issue = "생성된 답변이 문서에 기반하지 않았거나 너무 짧습니다. 혹은 생성된 답변에 중복된 내용이 포함되어 있습니다."

        retry_prompt = clarification_template.format(
            context=context_text,
            question=config["retriever"]["query"],
            issue=issue,
            chat_history=chat_history,
        )

        if model_type == "huggingface":
            answer = generate_answer_hf(retry_prompt, model_info, config["generator"])
        elif model_type == "openai":
            answer = generate_answer_openai(retry_prompt, model_info, config["generator"])
        else:
            raise ValueError(f"지원되지 않는 generator model_type: {model_type}")

        print(answer)

    if not is_unsatisfactory(answer):
        print("✅ 최종 답변 생성 완료")
    else:
        print("⚠️ 최대 재시도 횟수 도달, 답변 개선 실패")

    return answer
