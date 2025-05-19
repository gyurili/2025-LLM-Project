from typing import List
from langchain.schema import Document
from generator.load_model import load_generator_model, generate_answer
from generator.make_prompt import build_prompt_with_expansion, get_all_documents_from_vectorstore


def generator_main(
    retrieved_docs: List[Document],
    vectorstore,
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
    all_docs = get_all_documents_from_vectorstore(vectorstore) if vectorstore else []
    window = 1 if vectorstore else 0

    prompt = build_prompt_with_expansion(
        question=query,
        retrieved_docs=retrieved_docs,
        all_docs=all_docs,
        window=window,
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