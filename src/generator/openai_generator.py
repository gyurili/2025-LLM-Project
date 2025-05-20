import os
from typing import Dict
from langsmith import trace
import openai


def load_openai_model(config: Dict) -> Dict:
    """
    OpenAI 모델명을 받아 초기화합니다.
    """
    openai.api_key = os.getenv("OPENAI_API_KEY")
    return {"model": config["generator"]["model_name"]}


def generate_answer_openai(prompt: str, model_info: Dict, generation_config: Dict) -> str:
    """
    OpenAI ChatCompletion API를 사용하여 프롬프트에 응답을 생성합니다.
    """
    with trace(name="generate_answer_openai", inputs={"prompt": prompt}) as run:
        openai.api_key = os.getenv("OPENAI_API_KEY")

        response = openai.ChatCompletion.create(
            model=model_info["model"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=generation_config.get("max_length", 512),
            temperature=0.0
        )

        result = response["choices"][0]["message"]["content"]
        run.add_outputs({"output": result})
        return result