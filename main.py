from src.utils.shared_cache import set_cache_dirs
set_cache_dirs()

import os
import time
from typing import Dict
from langsmith import trace
from dotenv import load_dotenv
from src.loader.loader_main import loader_main
from src.embedding.embedding_main import embedding_main
from src.embedding.vector_db import generate_embedding
from src.retrieval.retrieval_main import retrieval_main
from src.utils.config import load_config
from src.utils.path import get_project_root_dir
from src.generator.generator_main import generator_main
from src.generator.chat_history import load_chat_history
from src.generator.hf_generator import load_hf_model
from src.generator.openai_generator import load_openai_model


'''
    TODO:
    - 각자 main수정에 맞게 generator_main, retrieval_main, embedding_main, loader_main 수정
    - 임베딩, 모델인포, 컨피그, dotenv등은 전역적으로 한번만 선언
'''
def get_generation_model(model_type: str, model_name: str, use_quantization: bool = False) -> Dict:
    """
    지정된 모델 타입 및 이름에 따라 생성 모델을 로드합니다.

    Args:
        model_type (str): 생성 모델 종류 ('huggingface' 또는 'openai')
        model_name (str): 사용할 모델 이름
        use_quantization (bool, optional): 양자화 사용 여부. 기본값은 False.

    Returns:
        Dict: 로드된 모델 정보 (예: pipeline, tokenizer 등 포함)
    """
    config = {
        'generator': {
            'model_type': model_type,
            'model_name': model_name,
            'use_quantization': use_quantization
        }
    }

    if model_type == 'huggingface':
        return load_hf_model(config)
    elif model_type == 'openai':
        return load_openai_model(config)
    else:
        raise ValueError(f"지원되지 않는 모델 타입: {model_type}")

def rag_pipeline(config, embeddings, model_info=None, is_save=False):
    try:
        with trace(name="rag_pipeline") as run:

            with trace(name="loader_main"):
                chunks = loader_main(config, embeddings)
                
            with trace(name="embedding_main"):
                vector_store = embedding_main(config, chunks, embeddings=embeddings, is_save=is_save)

            with trace(name="retrieval_main"):
                docs = retrieval_main(config, vector_store, chunks, embeddings=embeddings)

            with trace(name="generator_main"):
                start_time = time.time()
                answer = generator_main(docs, config, model_info=model_info)
                end_time = time.time()
                elapsed = round(end_time - start_time, 2)

            run.add_outputs({
                "query": config["retriever"]["query"],
                "model_type": config["generator"]["model_type"],
                "model_name": config["generator"]["model_name"],
                "max_length": config["generator"]["max_length"],
                "num_chunks": len(chunks),
                "num_retrieved_docs": len(docs),
                "answer_length": len(answer),
                "final_answer": answer
            })
            
            return docs, answer, elapsed
            
    except Exception as e:
        print(f"❌ 로깅 에러: {e}")

if __name__ == "__main__":
    project_root = get_project_root_dir()

    config_path = os.path.join(project_root, "config.yaml")
    dotenv_path = os.path.join(project_root, ".env")
    load_dotenv(dotenv_path=dotenv_path)

    config = load_config(project_root)

    embed_model_name = config["embedding"]["embed_model"]
    model_type = config["generator"]["model_type"]
    model_name = config["generator"]["model_name"]
    use_quantization = config["generator"]["use_quantization"]
    
    embeddings = generate_embedding(embed_model_name)
    model_info = get_generation_model(model_type, model_name, use_quantization)
    
    rag_pipeline(config, embeddings, model_info, is_save=True)
