from typing import List, Dict
from langchain.schema import Document
import torch


def load_generator_model(config: Dict) -> Dict:
    """
    주어진 config에 따라 다양한 유형의 언어 생성 모델을 초기화합니다.

    Args:
        config (Dict): 모델 구성 정보를 담은 설정 딕셔너리

    Returns:
        Dict: 모델 유형(type), tokenizer, model 등을 포함한 구조화 정보
    """
    model_type = config["model"]["type"]
    model_name = config["model"]["model_path"]

    if model_type == "huggingface":
        from transformers import AutoTokenizer, AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False,
            trust_remote_code=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # 또는 config 기반 설정도 가능
            device_map="auto",
            trust_remote_code=True
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
        model_info (Dict): 모델 로딩 결과 (type, tokenizer, model 등 포함)
        generation_config (Dict): temperature, max_tokens 등 생성 옵션

    Returns:
        str: 생성된 답변 문자열
    """
    if model_info["type"] == "hf":
        tokenizer = model_info["tokenizer"]
        model = model_info["model"]

        inputs = tokenizer(prompt, return_tensors="pt")

        # ✅ to(model.device) 적용은 input_ids와 attention_mask에만
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)

        generate_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": generation_config.get("max_tokens", 512),
            "temperature": generation_config.get("temperature", 0.7),
            "top_p": generation_config.get("top_p", 0.9),
        }

        # ✅ token_type_ids가 존재하는 경우에만 포함 (BERT 계열 지원)
        if "token_type_ids" in inputs:
            generate_kwargs["token_type_ids"] = inputs["token_type_ids"].to(model.device)

        with torch.no_grad():
            output = model.generate(**generate_kwargs)

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


def build_prompt_with_expansion(
    question: str,
    retrieved_docs: List[Document],
    all_docs: List[Document],
    window: int = 1,
    include_source: bool = True,
    prompt_template: str = None
) -> str:
    """
    검색된 문서 청크의 앞뒤 ±N 범위의 인접 청크까지 포함하여 프롬프트를 구성합니다.

    Args:
        question (str): 사용자 질문
        retrieved_docs (List[Document]): 검색된 핵심 청크 리스트
        all_docs (List[Document]): 전체 문서 청크 리스트 (검색 대상이 된 전체)
        window (int): 인접 청크 포함 범위 (기본 ±1)
        include_source (bool): 출처 포함 여부
        prompt_template (str): context와 question을 포함할 포맷 문자열

    Returns:
        str: 프롬프트 문자열
    """
    used_chunks = set()
    context_blocks = []

    for doc in retrieved_docs:
        file_name = doc.metadata.get("파일명")
        base_idx = doc.metadata.get("chunk_idx", 0)

        # 같은 문서 내 인접 청크 추출
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
            "다음은 RFP 문서를 기반으로 사용자의 질문에 답변하는 상황입니다.\n"
            "문서 내용을 기반으로만 답변하고, 모호하거나 문서에 없는 정보는 추측하지 마세요.\n\n"
            "[문서 정보]\n{context}\n\n"
            "[질문]\n{question}\n\n"
            "[답변]"
        )

    prompt = prompt_template.format(context=context_text, question=question)

    return prompt