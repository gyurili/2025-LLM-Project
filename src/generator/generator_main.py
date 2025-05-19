from typing import List
from langchain.schema import Document
from generator.load_model import load_generator_model, generate_answer
from generator.make_prompt import build_prompt


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

    # 2. 모델 로드
    model_info = load_generator_model(config)

    # 3. 답변 생성
    answer = generate_answer(
        prompt=prompt,
        model_info=model_info,
        generation_config=config.get("generator", {})
    )

    return answer