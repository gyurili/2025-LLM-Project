from typing import Dict
import torch


def load_generator_model(config: Dict) -> Dict:
    """
    주어진 config에 따라 다양한 유형의 언어 생성 모델을 초기화합니다.

    Args:
        config (Dict): 모델 구성 정보를 담은 설정 딕셔너리

    Returns:
        Dict: 모델 유형(type)과 모델 객체/토크나이저 등을 포함한 구조화된 정보
    """
    model_type = config["model"]["type"]
    model_name = config["model"]["model_path"]

    if model_type == "huggingface":
        from transformers import AutoTokenizer, AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model.eval()

        return {
            "type": "hf",
            "tokenizer": tokenizer,
            "model": model
        }

    elif model_type == "openai":
        import openai

        openai.api_key = config["model"].get("api_key", None)

        return {
            "type": "openai",
            "model": model_name
        }

    elif model_type == "dummy":
        return {
            "type": "mock"
        }

    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    


def generate_answer(
    prompt: str,
    model_info: Dict,
    generation_config: Dict
) -> str:
    """
    다양한 모델 유형에 따라 프롬프트에 대한 답변을 생성합니다.

    Args:
        prompt (str): 모델에 입력할 프롬프트 문자열
        model_info (Dict): 모델 로딩 결과 (타입, 객체 등 포함)
        generation_config (Dict): temperature, max_tokens 등 생성 옵션

    Returns:
        str: 생성된 답변 문자열
    """
    if model_info["type"] == "hf":
        tokenizer = model_info["tokenizer"]
        model = model_info["model"]

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=generation_config.get("max_tokens", 512),
                temperature=generation_config.get("temperature", 0.7),
                top_p=generation_config.get("top_p", 0.9)
            )

        return tokenizer.decode(output[0], skip_special_tokens=True)

    elif model_info["type"] == "openai":
        import openai

        response = openai.ChatCompletion.create(
            model=model_info["model"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=generation_config.get("max_tokens", 512),
            temperature=generation_config.get("temperature", 0.7),
            top_p=generation_config.get("top_p", 0.9)
        )

        return response["choices"][0]["message"]["content"]

    elif model_info["type"] == "mock":
        return "이건 테스트용 응답입니다."

    else:
        raise ValueError("Unsupported model type")
    
from typing import List, Dict


def build_prompt_with_expansion(
    question: str,
    retrieved_docs: List[Dict],
    all_docs: List[Dict],
    window: int = 1,
    include_source: bool = True,
    prompt_template: str = None
) -> str:
    """
    검색된 문서 청크의 앞뒤 ±N 범위의 인접 청크까지 포함하여 프롬프트를 구성합니다.

    Args:
        question (str): 사용자 질문
        retrieved_docs (List[Dict]): 검색된 핵심 청크 리스트
        all_docs (List[Dict]): 전체 문서 청크 리스트 (검색 대상이 된 전체)
        window (int): 인접 청크 포함 범위 (기본 ±1)
        include_source (bool): 출처 포함 여부
        prompt_template (str): context와 question을 포함할 포맷 문자열

    Returns:
        str: 프롬프트 문자열
    """
    used_chunks = set()
    context_blocks = []

    for doc in retrieved_docs:
        file_name = doc["source"]
        base_idx = doc["chunk_idx"]

        neighbors = [
            d for d in all_docs
            if d["source"] == file_name and
               abs(d["chunk_idx"] - base_idx) <= window
        ]

        for chunk in neighbors:
            key = (chunk["source"], chunk["chunk_idx"])
            if key in used_chunks:
                continue
            used_chunks.add(key)

            source_info = ""
            if include_source:
                source_info = (
                    f"[출처: {chunk['source']} | 기관: {chunk['기관']} | "
                    f"사업명: {chunk['사업명']}]"
                )

            context = f"{source_info}\n{chunk['text']}".strip()
            context_blocks.append(context)

    context_text = "\n\n---\n\n".join(context_blocks)

    if prompt_template is None:
        prompt_template = (
            "다음은 RFP 문서를 기반으로 사용자의 질문에 답변하는 상황입니다.\n"
            "문서 내용을 기반으로만 답변하고, 모호하거나 문서에 없는 정보는 추측하지 마세요.\n\n"
            "[문서 정보]\n{context}\n\n"
            "[질문]\n{question}\n\n"
            "[답변]"
        )

    prompt = prompt_template.format(context=context_text, question=question)

    return prompt