from typing import List
from langchain.schema import Document
from src.generator.hf_generator import load_hf_model, generate_answer_hf
from src.generator.openai_generator import load_openai_model, generate_answer_openai
from src.generator.make_prompt import build_prompt

def summarize_chat_history(config):
    """
    Langchain Document 또는 단순 문자열 기반 요약 생성
    """
    from src.generator.hf_generator import generate_answer_hf
    from src.generator.openai_generator import generate_answer_openai

    history_text = "\n".join(
        [f"{'질문' if turn['role'] == 'user' else '답변'}: {turn['content']}" for turn in config["chat_history"]]
    )
    prompt = f"다음은 사용자와 AI의 대화 내용입니다. 이 대화의 핵심 내용을 간결하게 요약해 주세요.\n\n{history_text}"

    if config["generator"]["model_type"] == "huggingface":
        model_info = load_hf_model(config)
        return generate_answer_hf(prompt, model_info, config["generator"])
    elif config["generator"]["model_type"] == "openai":
        model_info = load_openai_model(config)
        return generate_answer_openai(prompt, model_info, config["generator"])


def generator_main(
    retrieved_docs: List[Document],
    config: dict,
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
    # 1. 프롬프트 생성
    query = config["retriever"]["query"]

    # 2. 과거 질문, 답변 내역
    if config["chat_history"]:
        chat_history_str = "\n".join([f"질문: {turn['content']}" for turn in config["chat_history"]])

        # 3. 내역 요약
        chat_history_str = summarize_chat_history(config)
    else: # 대화 내역이 없을 경우
        chat_history_str = ""

    prompt = build_prompt(
        question=query,
        retrieved_docs=retrieved_docs,
        include_source=config.get("include_source", True),
        prompt_template=config.get("prompt_template"),
        chat_history=chat_history_str,
    )

    if config["generator"]["model_type"] == "huggingface":
        model_info = load_hf_model(config)
        answer = generate_answer_hf(prompt, model_info, config["generator"])

    elif config["generator"]["model_type"] == "openai":
        model_info = load_openai_model(config)
        answer = generate_answer_openai(prompt, model_info, config["generator"])
    
    print(answer)

    return answer