import os
import pandas as pd
from typing import List
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_teddynote.document_loaders import HWPLoader
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
import faiss

def data_load(path: str) -> pd.DataFrame:
    """
    주어진 경로에서 CSV 파일을 불러오고, 필요한 컬럼만 추출하여 전처리된 DataFrame을 반환합니다.

    Args:
        path (str): 불러올 CSV 파일의 상대 경로

    Returns:
        pd.DataFrame: 전처리된 데이터프레임

    Raises:
        FileNotFoundError: 파일이 존재하지 않을 경우
        pd.errors.ParserError: CSV 파싱 오류 발생 시
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_path = os.path.join(base_dir, path)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {full_path}")

    df = pd.read_csv(full_path)

    required_columns = ['파일명', '사업 요약', '텍스트', '사업명', '발주 기관', '사업 금액']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"필수 컬럼이 누락되었습니다: {set(required_columns) - set(df.columns)}")

    df = df[required_columns]
    df['사업 금액'] = pd.to_numeric(df['사업 금액'], errors='coerce').astype("Int64")
    return df

def data_process(df: pd.DataFrame) -> pd.DataFrame:
    """
    HWP 파일을 로드하여 각 행의 텍스트를 'full_text' 컬럼에 추가합니다.

    Args:
        df (pd.DataFrame): 전처리된 DataFrame

    Returns:
        pd.DataFrame: HWP 텍스트가 추가된 DataFrame
    """
    for file_name in df['파일명']:
        path = os.path.join("data/files", file_name)

        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"HWP 파일이 존재하지 않습니다: {path}")

            loader = HWPLoader(path)
            docs = loader.load()

            if docs and isinstance(docs[0].page_content, str):
                df.loc[df['파일명'] == file_name, 'full_text'] = docs[0].page_content
            else:
                print(f"파일 무시됨 (내용 없음): {file_name}")

        except Exception as e:
            print(f"파일 처리 오류 ({file_name}): {e}")
    return df

def hwp_chunking(df: pd.DataFrame) -> List[Document]:
    """
    각 문서 텍스트를 청크 단위로 나누고, metadata를 포함한 Document 리스트를 생성합니다.

    Args:
        df (pd.DataFrame): 텍스트가 포함된 DataFrame

    Returns:
        List[Document]: 문서 청크 리스트
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    all_chunks = []

    for _, row in df.iterrows():
        text = row.get('full_text')
        if isinstance(text, str) and text.strip():
            chunks = splitter.split_text(text)
            for i, chunk in enumerate(chunks):
                all_chunks.append(Document(
                    page_content=chunk,
                    metadata={
                        "사업명": row.get("사업명", ""),
                        "발주 기관": row.get("발주 기관", ""),
                        "파일명": row.get("파일명", ""),
                        "chunk_idx": i,
                    }
                ))
        else:
            print(f"텍스트 누락으로 스킵된 파일: {row.get('파일명')}")
    return all_chunks

def generate_vector_db(all_chunks: List[Document], embed_model_name: str) -> object:
    """
    문서 청크를 기반으로 FAISS 벡터 DB를 생성하고 저장합니다.

    Args:
        all_chunks (List[Document]): 문서 청크 리스트
        embed_model_name (str): 'open_ai' 또는 huggingface 모델 이름

    Returns:
        object: 생성된 embedding 객체

    Raises:
        ValueError: 지원하지 않는 임베딩 모델명인 경우
    """
    if embed_model_name == "open_ai":
        load_dotenv()
        embeddings = OpenAIEmbeddings()
    else:
        try:
            embeddings = HuggingFaceEmbeddings(model_name=embed_model_name)
        except Exception as e:
            raise ValueError(f"임베딩 모델 초기화 실패: {e}")

    dimension = len(embeddings.embed_query("hello world"))

    vector_store = FAISS(
        embedding_function=embeddings,
        index=faiss.IndexFlatL2(dimension),
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    vector_store.add_documents(all_chunks)
    vector_store.save_local(folder_path="data/", index_name="hwp_faiss_index")
    return embeddings

def load_vector_db(path: str, embeddings) -> FAISS:
    """
    저장된 FAISS 벡터 DB를 로드합니다.

    Args:
        path (str): 벡터 DB가 저장된 경로
        embeddings: 사용할 임베딩 모델 객체

    Returns:
        FAISS: 불러온 벡터 스토어
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"벡터 DB 경로가 존재하지 않습니다: {path}")

    return FAISS.load_local(
        folder_path=path,
        index_name="hwp_faiss_index",
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )

df = data_load("data/data_list.csv")
df = data_process(df)
all_chunks = hwp_chunking(df)
print("Chunking 완료!")
embeddings = generate_vector_db(all_chunks, "open_ai")
print("Vector DB가 저장되었습니다!")

print("저장된 Vector DB를 불러옵니다!")
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
vector_db_path = os.path.join(base_dir, "data")

vector_store = load_vector_db(vector_db_path, embeddings)

docs = vector_store.similarity_search("배드민턴장 및 탁구장 예약방법", k=3)

for i in range(len(docs)):
    print(docs.page_content)