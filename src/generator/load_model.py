import os
import torch
from typing import Dict
from inspect import signature
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langsmith import trace


def load_generator_model(config: Dict) -> Dict:
    """
    다양한 유형의 언어 생성 모델을 config를 기반으로 초기화합니다.

    Args:
        config (Dict): 모델 설정 정보

    Returns:
        Dict: {type, tokenizer, model} 형태의 딕셔너리

    To Do:
        - 모델 캐시 경로 지정 옵션 추가
        - 토크나이저 padding 관련 설정 명시화
    """
    model_type = config["generator"]["model_type"]
    model_name = config["generator"]["model_name"]

    if model_type == "huggingface":
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=False,
                trust_remote_code=True,
                token=os.getenv("HF_TOKEN")
            )
        except Exception as e:
            raise RuntimeError(f"❌ (generator.load_model.load_generator_model) 토크나이저 로딩 실패: {e}") 

        try:
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
        except Exception as e:
            raise RuntimeError(f"❌ (generator.load_model.load_generator_model) 모델 로딩 실패: {e}") 

        model.eval()
        return {"type": "hf", "tokenizer": tokenizer, "model": model}

    elif model_type == "openai":
        try:
            import openai
            openai.api_key = os.getenv("OPENAI_API_KEY")
            if not openai.api_key:
                raise ValueError("❌ (generator.load_model.load_generator_model) OPENAI_API_KEY가 설정되지 않았습니다.")
            return {"type": "openai", "model": model_name}
        except ImportError:
            raise ImportError("❌ (generator.load_model.load_generator_model) openai 패키지가 설치되지 않았습니다.")

    elif model_type == "dummy":
        return {"type": "mock"}

    else:
        raise ValueError(f"❌ (generator.load_model.load_generator_model) 지원하지 않는 model_type: {model_type}")


def generate_answer(prompt: str, model_info: Dict, generation_config: Dict) -> str:
    """
    주어진 프롬프트를 기반으로 다양한 언어 모델로부터 답변을 생성합니다.
    LangSmith 수동 트레이싱을 통해 실행 기록을 명시적으로 남깁니다.

    Args:
        prompt (str): 생성할 프롬프트
        model_info (Dict): 모델 및 토크나이저 정보
        generation_config (Dict): 생성 관련 설정 (max_length 등)

    Returns:
        str: 생성된 텍스트 응답

    TODO:
        - 응답 후처리 필터 추가
        - 출력 문자열 정제 옵션 추가
    """
    if not prompt:
        raise ValueError("❌ (generator.load_model.generate_answer) prompt가 비어 있습니다.")
    
    if not isinstance(model_info, dict) or "type" not in model_info:
        raise ValueError("❌ (generator.load_model.generate_answer) model_info의 형식이 잘못되었거나 'type' 키가 없습니다.")
    
    with trace(name="generate_answer", inputs={"prompt": prompt}) as run:
        if model_info["type"] == "hf":
            try:
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

                raw_output = tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                answer = raw_output.strip()

                bad_tokens = ["하십시오", "하실 수", "알고 싶어요", "하는데 필요한", "것을", "한다", "하십시오.", "하시기 바랍니다"]
                for token in bad_tokens:
                    answer = answer.replace(token, "")

                if len(answer) < 10 or answer.count(" ") < 3:
                    answer = "해당 문서에서 예약 방법에 대한 명확한 정보를 찾을 수 없습니다."

                run.add_outputs({"output": answer})
                return answer
            except Exception as e:
                raise RuntimeError(f"❌ (generator.load_model.generate_answer) HuggingFace 모델 응답 생성 중 오류: {e}")

        elif model_info["type"] == "openai":
            try:
                import openai
                openai.api_key = os.getenv("OPENAI_API_KEY")
                if not openai.api_key:
                    raise ValueError("❌ (generator.load_model.generate_answer) OPENAI_API_KEY가 설정되지 않았습니다.")

                response = openai.ChatCompletion.create(
                    model=model_info["model"],
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=generation_config.get("max_length", 512),
                    temperature=0.0
                )
                result = response["choices"][0]["message"]["content"]
                run.add_outputs({"output": result})
                return result
            except Exception as e:
                raise RuntimeError(f"❌ (generator.load_model.generate_answer) OpenAI 모델 호출 오류: {e}")

        elif model_info["type"] == "mock":
            result = "이건 테스트용 응답입니다."
            run.add_outputs({"output": result})
            return result

        else:
            raise ValueError(f"❌ (generator.load_model.generate_answer) 지원하지 않는 model_info['type']: {model_info['type']}")