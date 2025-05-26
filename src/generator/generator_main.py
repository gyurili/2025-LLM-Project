from typing import List
from langchain.schema import Document

from src.generator.hf_generator import generate_answer_hf
from src.generator.openai_generator import generate_answer_openai
from src.generator.make_prompt import build_prompt
from src.generator.chat_history import load_chat_history

'''
    TODO:
    - 변경된 모듈들에 맞춰서 generator_main 함수 수정 ✅
    - load_chat_history 함수 위치 변경에 의한, import 경로 수정 ✅
'''

def generator_main(
    retrieved_docs:List[Document],
    config:dict,
    model_info:dict = None
) -> str:
    """
    검색된 문서 리스트를 기반으로 답변을 생성하는 메인 실행 함수.

    Args:
        retrieved_docs (List[Document]): 검색된 핵심 청크 리스트
        config (dict): 설정 정보 (모델, 템플릿 등 포함)
        model_info (dict): 모델 정보 (Hugging Face 또는 OpenAI 모델 정보)

    Returns:
        str: 생성된 답변
    """
    if model_info is None:
        raise ValueError("❌ (generator.generator_main.generator_main) model_info가 없습니다.")

    chat_history = load_chat_history(config)

    query = config["retriever"]["query"]

    prompt = build_prompt(
        question=query,
        retrieved_docs=retrieved_docs,
        include_source=config.get("include_source", True),
        prompt_template=config.get("prompt_template"),
        chat_history=chat_history,
    )

    model_type = config["generator"]["model_type"]
            
    if model_type == "huggingface":
        answer = generate_answer_hf(prompt, model_info, config["generator"])
    elif model_type == "openai":
        answer = generate_answer_openai(prompt, model_info, config["generator"])

    print(f"✅ 과거 맥락 :{chat_history}")
    print(f"✅ 답변 :{answer}")

    
    print(answer)
    print("✅ 답변 생성 완료")
    return answer