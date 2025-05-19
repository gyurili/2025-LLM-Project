import os
import torch
from typing import Dict
from inspect import signature
from langchain_community.llms import OpenAI
from langchain_huggingface import HuggingFacePipeline
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_core.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langsmith import traceable


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


@traceable(name="generate_answer")
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