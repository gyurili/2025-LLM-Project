from typing import List
from langchain.schema import Document
from src.generator.hf_generator import load_hf_model, generate_answer_hf
from src.generator.openai_generator import load_openai_model, generate_answer_openai
from src.generator.make_prompt import build_prompt


def generator_main(
    retrieved_docs: List[Document],
    config: dict
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

    prompt = build_prompt(
        question=query,
        retrieved_docs=retrieved_docs,
        include_source=config.get("include_source", True),
        prompt_template=config.get("prompt_template")
    )

    if config["generator"]["model_type"] == "huggingface":
        model_info = load_hf_model(config)
        answer = generate_answer_hf(prompt, model_info, config["generator"])

    elif config["generator"]["model_type"] == "openai":
        model_info = load_openai_model(config)
        answer = generate_answer_openai(prompt, model_info, config["generator"])
    
    print(answer)

    return answer