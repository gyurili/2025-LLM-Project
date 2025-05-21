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
    print("✅ 답변 생성 완료")

    return answer

def is_unsatisfactory(answer: str) -> bool:
    '''
    생성된 답변이 적절한지 판단하는 간단한 규칙 기반 평가 함수.

    Args:
        answer: 생성 모델에서 프롬프트 기반으로 생성된 답변

    Returns:
        bool: 생성된 답변이 적절하지 않다면 True, 괜찮다면 False
    '''
    if '정보가 없습니다' in answer or '잘 모르겠습니다' in answer:
        return True
    if len(answer.strip()) < 10:
        return True
    return False

def generate_with_clarification(
    retrieved_docs: List[Document], 
    config: dict,
    max_retries:int=2
) -> str:
    '''
    '''
    answer = generator_main(retrieved_docs, config)
    
    # 리트라이 루프
    for _ in range(max_retries):
        if not is_unsatisfactory(answer):
            break
        print("⚠️ 답변 재생성 중...")
        issue = "답변이 너무 짧거나 문서의 정보를 기반으로 하지 않았습니다."
        clarification_template = (
            "이전 답변은 다음 문제로 인해 부적절했습니다: {issue}\n"
            "문서 내용에 명시되지 않은 정보를 유추하지 말고, 명확한 문서 기반 답변을 생성하세요.\n\n"
            "### 문서 내용:\n{context}\n\n"
            "### 질문:\n{question}\n\n"
            "### 개선된 답변:"
        )

        retry_prompt = clarification_template.format(
            context="\n\n---\n\n".join([d.page_content for d in retrieved_docs]),
            question=config['retriever']['query'],
            issue=issue
        )

        if config["generator"]["model_type"] == "huggingface":
            model_info = load_hf_model(config)
            answer = generate_answer_hf(retry_prompt, model_info, config["generator"])
        elif config["generator"]["model_type"] == "openai":
            model_info = load_openai_model(config)
            answer = generate_answer_openai(retry_prompt, model_info, config["generator"])

    print(answer)
    print("✅ 최종 답변 생성 완료")

    return answer