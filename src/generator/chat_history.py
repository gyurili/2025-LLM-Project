from src.generator.hf_generator import generate_answer_hf
from src.generator.openai_generator import generate_answer_openai


def summarize_chat_history(config, model_info=None):
    """
    질의응답 내역 요약을 생성하는 함수

    Args:
        config (dict): 설정 정보 (모델, 템플릿 등 포함)
        model_info (dict, optional): 모델 정보 (Hugging Face 또는 OpenAI 모델 정보)

    Returns:
        str: 요약된 대화 내역
    """
    if model_info is None:
        raise ValueError("❌ (generator.chat_history.summarize_chat_history) model_info가 없습니다.")

    history_text = "\n".join(
        [f"{'질문' if turn['role'] == 'user' else '답변'}: {turn['content']}" for turn in config["chat_history"]]
    )
    prompt = f"다음은 사용자와 AI의 대화 내용입니다. 이 대화의 핵심 내용을 간결하게 요약해 주세요.\n\n{history_text}"

    if config["generator"]["model_type"] == "huggingface":
        return generate_answer_hf(prompt, model_info, config["generator"])
    elif config["generator"]["model_type"] == "openai":
        return generate_answer_openai(prompt, model_info, config["generator"])
    

def load_chat_history(config, model_info=None):
    """
    질의응답 내역 요약을 로드하는 함수
    이전 질의응답 내역이 없는 경우, 빈 문자열을 반환

    Args:
        config (dict): 설정 정보 (모델, 템플릿 등 포함)

    Returns:
        str: 요약된 대화 내역 or 빈 문자열
    """
    if config["chat_history"]:
        chat_history_str = "\n".join([f"질문: {turn['content']}" for turn in config["chat_history"]])

        chat_history_str = summarize_chat_history(config, model_info)
        print(f"과거 대화 내역 요약: {chat_history_str}")
        return chat_history_str
    else:
        chat_history_str = ""
        return chat_history_str

