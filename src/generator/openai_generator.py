import os
from typing import Dict
from langsmith import trace
from openai import OpenAI

'''
    TODO:
    - OpenAI 모델 로딩 시 API 키 확인 ✅
    - 답변 생성 유지 ✅
'''


def load_openai_model(config: Dict) -> Dict:
    """
    OpenAI 모델명을 받아 초기화합니다.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("❌ OPENAI_API_KEY가 설정되어 있지 않습니다.")

    client = OpenAI(api_key=api_key)

    return {
        "type": "openai",
        "model": config["generator"]["model_name"],
        "client": client
    }


def generate_answer_openai(prompt: str, model_info: Dict, generation_config: Dict) -> str:
    """
    OpenAI ChatCompletion API를 사용하여 프롬프트에 응답을 생성합니다.
    """
    with trace(name="generate_answer_openai", inputs={"prompt": prompt}) as run:
        client = model_info["client"]
        model = model_info["model"]

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=generation_config.get("max_length", 512),
            temperature=0.0,
        )

        answer = response.choices[0].message.content.strip()
        
        # 후처리 필터
        bad_tokens = ["하십시오", "하실 수", "알고 싶어요", "하는데 필요한", "것을", "한다", "하십시오.", "하시기 바랍니다"]
        for token in bad_tokens:
            answer = answer.replace(token, "")

        if len(answer) < 10 or answer.count(" ") < 3:
            answer = "해당 문서에서 예약 방법에 대한 명확한 정보를 찾을 수 없습니다."

        run.add_outputs({"output": answer})
        return answer