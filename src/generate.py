import os
import torch
from typing import List
from langchain.docstore.document import Document
from langchain_community.llms import OpenAI
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_core.prompts import PromptTemplate
import yaml

def initialize_llm(config: dict):
    """LLM을 초기화하는 함수"""
    if config["llm"]["provider"] == "huggingface":

        # Hugging Face 모델 이름
        language_model_name = config["llm"]["model"]
        print(f"LLM 모델 이름: {language_model_name}")

        # LLM 모델 이름이 설정되지 않았는지 확인
        if not language_model_name:
            raise ValueError("❌ LLM 모델 이름이 설정되지 않았습니다.")

        # 양자화 사용 여부부
        if config["llm"]["use_quantization"] == True:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                llm_int8_enable_fp32_cpu_offload=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                language_model_name,
                quantization_config=bnb_config,
                trust_remote_code=True,
                token=os.getenv("HF_TOKEN"),
                device_map="auto",
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                language_model_name,
                trust_remote_code=True,
                token=os.getenv("HF_TOKEN"),
                device_map="auto",
            )

        tokenizer = AutoTokenizer.from_pretrained(
            language_model_name,
            token=os.getenv("HF_TOKEN"),
            )
        print("✅ 모델과 토크나이저 로드 완료")

        llm_pipeline = pipeline(
            model=model,
            tokenizer=tokenizer,
            task="text-generation",
            do_sample=True,
            temperature=0.2,
            repetition_penalty=1.2,
            return_full_text=False,
            max_new_tokens=1000,
        )
        print("✅ 파이프라인 생성 완료")
        
        llm = HuggingFacePipeline(pipeline=llm_pipeline)
        print(llm)

        # chat_model = ChatHuggingFace(llm=llm)
        # print("✅ Chat 모델 생성 완료")

    elif config["llm"]["provider"] == "openai":
        # OpenAI API는 별도 초기화 불필요
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("❌ OpenAI API 키가 설정되지 않았습니다.")
    else:
        raise ValueError("❌ 지원하지 않는 LLM provider입니다.")

    return llm # , chat_model

def generate_answer(retriever, docs: List[Document], query: str, llm, config: dict) -> str:
    """검색된 문서와 쿼리를 기반으로 답변을 생성하는 함수"""
    try:
        # 검색된 문서의 내용을 결합
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # 프롬프트 구성
        template = config["llm"]["prompt_template"]
        prompt = PromptTemplate.from_template(template)

        # LLM 호출
        # if config["llm"]["provider"] == "huggingface":
        #     retrieval_chain = (
        #         {"context": retriever | format_docs, "question": RunnablePassthrough()}
        #         | prompt
        #         | llm
        #         | StrOutputParser()

        # LLM 호출 Retriever가 아닌, Docs를 직접 넘기기
        if config["llm"]["provider"] == "huggingface":
            retrieval_chain = (
                {"context": lambda _: format_docs(docs), "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
)
        elif config["llm"]["provider"] == "openai":
            # OpenAI API 예시
            # client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            # response = client.chat.completions.create(
            #     model=config["llm"]["model"],
            #     messages=[{"role": "user", "content": prompt}],
            #     max_tokens=config["llm"]["max_tokens"],
            # )
            # answer = response.choices[0].message.content
            pass
        else:
            raise ValueError("❌ 지원하지 않는 LLM provider입니다.")

        return retrieval_chain.invoke(query)
    
    except Exception as e:
        print(f"❌ 답변 생성 오류: {e}")
        return "⚠️ 답변 생성 중 오류가 발생했습니다."
    