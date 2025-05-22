import os
import fitz
import easyocr
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from typing import List
from langchain.schema import Document
from langchain_teddynote.document_loaders import HWPLoader
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers.util import cos_sim
from src.utils.path import get_project_root_dir

# EasyOCR Reader 객체 생성 (GPU 사용)
# 한글(ko) + 영어(en)를 인식하며, 모델은 한 번만 로드됨
reader = easyocr.Reader(['ko', 'en'], gpu=True)


def safe_ocr(img_array: np.ndarray, ocr_reader: easyocr.Reader) -> str:
    """
    이미지 배열을 입력받아 EasyOCR로 텍스트를 추출합니다.

    Args:
        img_array (np.ndarray): OCR을 수행할 이미지 배열
        ocr_reader (easyocr.Reader): 초기화된 EasyOCR 리더 인스턴스

    Returns:
        str: 추출된 텍스트 문자열
    """
    try:
        result = ocr_reader.readtext(img_array, detail=0)
        if not isinstance(result, list):
            return ""
        return "\n".join(result)
    except Exception as e:
        raise RuntimeError(f"❌ [OCR] (data_loader.safe_ocr) OCR 처리 실패: {e}")


def extract_text_from_pdf(pdf_path: Path, apply_ocr: bool = True) -> str:
    """
    PDF 파일에서 텍스트 및 OCR 텍스트를 추출합니다.

    Args:
        pdf_path (Path): 처리할 PDF 파일 경로
        apply_ocr (bool): OCR 수행 여부

    Returns:
        str: 전체 페이지에서 추출된 텍스트
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"❌(data_loader.extract_text_from_pdf.pdf_path) PDF 파일을 찾을 수 없습니다: {pdf_path}")
    
    full_text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(tqdm(doc, desc=f"{pdf_path.name}")):
                try:
                    page_text = page.get_text()
                    full_text += page_text

                    if apply_ocr:
                        pix = page.get_pixmap(dpi=300)
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        ocr_text = safe_ocr(np.array(img), reader)
                        if ocr_text.strip():
                            full_text += f"\n[OCR p.{page_num + 1}]\n{ocr_text}"
                except Exception as e:
                    print(f"⚠️ [Warning] (data_loader.extract_text_from_pdf) 페이지 {page_num + 1} 처리 중 오류: {e}")
    except Exception as e:
        raise RuntimeError(f"❌ [Runtime] (data_loader.extract_text_from_pdf) PDF 파일 처리 오류: {e}")
    
    return full_text


def retrieve_top_documents_from_metadata(query, csv_path, embed_model, top_k=5, verbose=False):
    """
    사용자 질문(query)과 문서 메타데이터(csv)에 기반하여 
    가장 유사한 top_k개의 문서를 반환합니다.

    Parameters:
        query (str): 사용자 질문
        csv_path (str): CSV 파일 경로
        embed_model (str): 임베딩 모델 이름 (예: "openai", "huggingface")
        top_k (int): 반환할 문서 수 (기본값 5)
        verbose (bool): 결과를 표 형태로 출력할지 여부 (기본값 False)

    Returns:
        pd.DataFrame: 상위 top_k 문서 정보 + 유사도 점수
    """
    try:  # 수정부분: 전체 함수 방어적 처리 시작
        from src.embedding.vector_db import generate_embedding
        embedder = generate_embedding(embed_model)
        if embedder is not None:
            if verbose:
                print(f"    📌 [Info] Embedding model: {embedder.__class__.__name__}")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"❌ (loader.data_loader.retrieve_top_documents_from_metadata) 파일을 찾을 수 없습니다: {csv_path}")

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            raise ValueError(f"❌ (loader.data_loader.retrieve_top_documents_from_metadata) CSV 파일 로딩 실패: {str(e)}")

        required_columns = ["사업명", "발주 기관", "사업 요약", "파일명"]
        for col in required_columns:
            if col not in df.columns:
                raise KeyError(f"❌ (loader.data_loader.retrieve_top_documents_from_metadata) '{col}' 열이 CSV에 존재하지 않습니다.")

        def make_embedding_text(row):
            return f"{row['사업명']} {row['발주 기관']} {row['사업 요약']}"

        try:
            df["임베딩텍스트"] = df.apply(make_embedding_text, axis=1)
        except Exception as e:
            raise RuntimeError(f"❌ (loader.data_loader.retrieve_top_documents_from_metadata) 임베딩 텍스트 생성 중 오류: {str(e)}")

        doc_texts = df["임베딩텍스트"].tolist()

        if hasattr(embedder, "encode"):
            doc_embeddings = embedder.encode(doc_texts, convert_to_tensor=True)
            query_embedding = embedder.encode(query, convert_to_tensor=True)
            similarities = cos_sim(query_embedding, doc_embeddings)[0].cpu().numpy()
        else:
            doc_embeddings = embedder.embed_documents(doc_texts)
            query_embedding = embedder.embed_query(query)
            similarities = cosine_similarity(
                np.array([query_embedding]), np.array(doc_embeddings)
            )[0]

        top_k_indices = np.argsort(similarities)[::-1][:top_k]

        try:
            top_docs = df.iloc[top_k_indices].copy()
            top_docs["유사도"] = similarities[top_k_indices]
        except Exception as e:
            raise RuntimeError(f"❌ (loader.data_loader.retrieve_top_documents_from_metadata) 결과 DataFrame 생성 실패: {str(e)}")

        if verbose == True:
            from tabulate import tabulate
            table = [
                [idx, row["파일명"], f"{row['유사도']:.4f}"]
                for idx, row in top_docs.iterrows()
            ]
            output = tabulate(table, headers=["IDX", "파일명", "유사도"], tablefmt="github")
            print("\n".join("    " + line for line in output.splitlines()))  # 수정부분: 4칸 들여쓰기 적용

        return top_docs
    except Exception as e:
        raise RuntimeError(f"❌ (loader.data_loader.retrieve_top_documents_from_metadata) 예외 발생: {e}")  # 수정부분: 전체 함수 방어적 처리 끝

from src.utils.path import get_project_root_dir

def data_process(df: pd.DataFrame, apply_ocr: bool = True, file_type: str = "all") -> pd.DataFrame:
    """
    HWP 또는 PDF 파일을 처리하여 텍스트를 추출하고 full_text 컬럼에 저장합니다.

    Args:
        df (pd.DataFrame): 파일 목록을 포함한 데이터프레임
        apply_ocr (bool): PDF OCR 여부
        file_type (str): 'hwp', 'pdf', 'all' 중 하나

    Returns:
        pd.DataFrame: 텍스트가 추가된 DataFrame
    """
    base_dir = get_project_root_dir()
    file_root = os.path.join(base_dir, "data", "files")

    if file_type in ["hwp", "pdf"]:
        mask = df['파일명'].str.lower().str.endswith(f".{file_type}")
        filtered_df = df[mask].copy()
    elif file_type == "all":
        filtered_df = df.copy()

    filtered_df['full_text'] = None

    for file_name in filtered_df['파일명']:
        file_path = os.path.join(file_root, file_name)
        try:
            if not os.path.exists(file_path):
                 raise FileNotFoundError(f"❌ [FileNotFound] (data_loader.data_process.path) 파일이 존재하지 않습니다: {file_path}")

            if file_name.lower().endswith(".hwp") and file_type in ["hwp", "all"]:
                loader = HWPLoader(file_path)
                docs = loader.load()
                if docs and isinstance(docs[0].page_content, str):
                    filtered_df.loc[filtered_df['파일명'] == file_name, 'full_text'] = docs[0].page_content
                else:
                    print(f"⚠️ [Warning] (data_loader.data_process.hwp) HWP 파일 무시됨 (내용 없음): {file_name}")

            elif file_name.lower().endswith(".pdf") and file_type in ["pdf", "all"]:
                text = extract_text_from_pdf(Path(file_path), apply_ocr=apply_ocr)
                filtered_df.loc[filtered_df['파일명'] == file_name, 'full_text'] = text

            else:
                print(f"⚠️ [Warning] (data_loader.data_process) 지원하지 않는 파일 형식입니다: {file_name}")

        except Exception as e:
            raise RuntimeError(f"❌ [Runtime] (data_loader.data_process) 파일 처리 오류 ({file_name}): {e}")  

    return filtered_df.reset_index(drop=True)