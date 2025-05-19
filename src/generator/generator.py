import os
import torch
from typing import List, Dict
from inspect import signature
from langchain.schema import Document
from langchain.vectorstores.faiss import FAISS
from langchain_community.llms import OpenAI
from langchain_huggingface import HuggingFacePipeline
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_core.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_core.tracers import tracing_v2_enabled


def load_generator_model(config: Dict) -> Dict:
    """
    다양한 유형의 언어 생성 모델을 config를 기반으로 초기화합니다.

    Args:
        config (Dict): 모델 설정 정보

    Returns:
        Dict: {type, tokenizer, model} 형태의 딕셔너리
    """
    model_type = config["generator"]["model_type"]
    model_name = config["generator"]["model_name"]

    if model_type == "huggingface":
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False,
            trust_remote_code=True,
            token=os.getenv("HF_TOKEN")
        )

        if config["generator"].get("use_quantization", False):
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                llm_int8_enable_fp32_cpu_offload=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                trust_remote_code=True,
                token=os.getenv("HF_TOKEN"),
                device_map="auto"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                token=os.getenv("HF_TOKEN"),
                device_map="auto"
            )

        model.eval()
        return {"type": "hf", "tokenizer": tokenizer, "model": model}

    elif model_type == "openai":
        import openai
        openai.api_key = os.getenv("OPENAI_API_KEY")
        return {"type": "openai", "model": model_name}

    elif model_type == "dummy":
        return {"type": "mock"}

    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def generate_answer(prompt: str, model_info: Dict, generation_config: Dict) -> str:
    """
    주어진 프롬프트를 기반으로 다양한 언어 모델로부터 답변을 생성합니다.
    LangSmith 트레이서를 통해 실행 기록을 남깁니다.

    Args:
        prompt (str): 생성할 프롬프트
        model_info (Dict): 모델 및 토크나이저 정보
        generation_config (Dict): 생성 관련 설정 (max_length 등)

    Returns:
        str: 생성된 텍스트 응답
    """
    with tracing_v2_enabled(project_name="RAG-Generator"):
        if model_info["type"] == "hf":
            tokenizer = model_info["tokenizer"]
            model = model_info["model"]

            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs["attention_mask"].to(model.device)

            generate_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": generation_config.get("max_length", 512),
                "do_sample": False,
                "eos_token_id": tokenizer.eos_token_id or tokenizer.pad_token_id,
                "repetition_penalty": 1.2
            }

            generate_signature = signature(model.generate).parameters
            if "token_type_ids" in inputs and "token_type_ids" in generate_signature:
                generate_kwargs["token_type_ids"] = inputs["token_type_ids"].to(model.device)

            with torch.no_grad():
                output = model.generate(**generate_kwargs)

            return tokenizer.decode(output[0], skip_special_tokens=True)

        elif model_info["type"] == "openai":
            import openai
            openai.api_key = os.getenv("OPENAI_API_KEY")
            response = openai.ChatCompletion.create(
                model=model_info["model"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=generation_config.get("max_length", 512),
                temperature=0.0
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
    검색된 문서 청크의 앞뒤 ±N 범위 인접 청크를 포함해 프롬프트를 구성합니다.

    Args:
        question (str): 사용자 질문
        retrieved_docs (List[Document]): 검색된 핵심 청크들
        all_docs (List[Document]): 전체 청크 리스트
        window (int): 인접 청크 포함 범위
        include_source (bool): 출처 포함 여부
        prompt_template (str): 사용자 정의 프롬프트 템플릿

    Returns:
        str: 구성된 프롬프트 문자열
    """
    used_chunks = set()
    context_blocks = []

    for doc in retrieved_docs:
        file_name = doc.metadata.get("파일명")
        base_idx = doc.metadata.get("chunk_idx", 0)

        if window == 0:
            neighbors = [doc]
        else:
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
            "아래는 정부 및 대학 관련 사업 문서 요약입니다. 문서 내용을 바탕으로 다음 질문에 대해 명확하고 핵심적으로 답변하세요.\n"
            "반복하지 말고, 문서에 포함된 정보만 기반으로 답하세요. 문서에 없는 정보는 '해당 문서에 정보가 없습니다.'라고 하세요.\n\n"
            "### 문서 내용:\n{context}\n\n"
            "### 질문:\n{question}\n\n"
            "### 답변 (문장으로 1~3줄):"
        )

    return prompt_template.format(context=context_text, question=question)


def get_all_documents_from_vectorstore(vectorstore: FAISS) -> List[Document]:
    """
    FAISS 벡터스토어에서 전체 Document 리스트를 반환합니다.

    Args:
        vectorstore (FAISS): 벡터스토어 객체

    Returns:
        List[Document]: 전체 문서 청크 리스트
    """
    if not hasattr(vectorstore, "docstore") or not hasattr(vectorstore.docstore, "_dict"):
        raise ValueError("해당 벡터스토어는 docstore 정보를 제공하지 않습니다.")

    return list(vectorstore.docstore._dict.values())