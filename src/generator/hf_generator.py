import os
import re
import torch
from typing import Dict
from inspect import signature
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langsmith import trace


def load_hf_model(config: Dict) -> Dict:
    """
    Hugging Face 모델을 config를 기반으로 초기화합니다.
    """
    model_name = config["generator"]["model_name"]

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
    return {"tokenizer": tokenizer, "model": model}


def generate_answer_hf(prompt: str, model_info: Dict, generation_config: Dict) -> str:
    """
    Hugging Face 모델을 사용하여 프롬프트에 응답을 생성합니다

    Args:
        prompt (str): 생성에 사용할 프롬프트
        model_info (Dict): 'tokenizer', 'model' 키 포함
        generation_config (Dict): max_length 등 생성 설정

    Returns:
        str: 정제된 모델 응답

    To Do:
        - 불필요한 반복 제거
        - 프롬프트 누락 응답 제거
    """
    with trace(name="generate_answer_hf", inputs={"prompt": prompt}) as run:
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

        if "token_type_ids" in inputs and "token_type_ids" in signature(model.generate).parameters:
            generate_kwargs["token_type_ids"] = inputs["token_type_ids"].to(model.device)

        with torch.no_grad():
            output = model.generate(**generate_kwargs)

        raw_output = tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        answer = raw_output.strip()

        # 프롬프트 시작 문장 제거
        if "당신은 정부 및 대학의 공공 사업 제안서를 분석하는" in answer:
            answer = answer.split("문서 내용:")[-1].strip()

        # 반복 제거
        bad_tokens = ["하십시오", "하실 수", "알고 싶어요", "하는데 필요한", "것을", "한다", "하십시오.", "하시기 바랍니다"]
        for token in bad_tokens:
            answer = answer.replace(token, "")

        # 너무 짧거나 의미 없는 경우 대체
        if len(answer) < 10 or answer.count(" ") < 3:
            answer = "해당 문서에서 질문에 대한 명확한 정보를 찾을 수 없습니다"

        run.add_outputs({"output": answer})
        return answer