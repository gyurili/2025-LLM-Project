import os
import easyocr
import fitz  # PyMuPDF
import numpy as np
import pandas as pd

from pathlib import Path
from PIL import Image
from tqdm import tqdm
from langchain_teddynote.document_loaders import HWPLoader
from tabulate import tabulate
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers.util import cos_sim


def safe_ocr(img_array: np.ndarray, ocr_reader: easyocr.Reader) -> str:
    """
    EasyOCR를 이용해 이미지 배열에서 텍스트를 추출합니다.

    Args:
        img_array (np.ndarray): OCR을 수행할 이미지 배열
        ocr_reader (easyocr.Reader): EasyOCR 리더 인스턴스

    Returns:
        str: 추출된 텍스트 문자열

    Raises:
        RuntimeError: OCR 처리 실패 시 발생
    """
    try:
        result = ocr_reader.readtext(img_array, detail=0)
        if not isinstance(result, list):
            return ""
        return "\n".join(result)
    except Exception as e:
        raise RuntimeError(f"❌ [Runtime] (data_loader.safe_ocr) OCR 처리 실패: {e}")


def extract_text_from_pdf(pdf_path: Path, apply_ocr: bool = True) -> str:
    """
    PDF 파일에서 텍스트를 추출하고, 필요시 OCR 결과도 병합합니다.

    Args:
        pdf_path (Path): PDF 파일 경로
        apply_ocr (bool): OCR 적용 여부

    Returns:
        str: 모든 페이지의 텍스트가 병합된 문자열

    Raises:
        FileNotFoundError: PDF 파일이 존재하지 않을 때
        RuntimeError: PDF 페이지 또는 전체 파일 처리 실패 시
    """
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"❌ [FileNotFound] (data_loader.extract_text_from_pdf.pdf_path) PDF 파일을 찾을 수 없습니다: {pdf_path}"
        )

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
                        if not hasattr(extract_text_from_pdf, "reader"):
                            import torch
                            gpu_available = torch.cuda.is_available()
                            extract_text_from_pdf.reader = easyocr.Reader(['ko', 'en'], gpu=gpu_available)
                        ocr_text = safe_ocr(np.array(img), extract_text_from_pdf.reader)
                        if ocr_text.strip():
                            full_text += f"\n[OCR p.{page_num + 1}]\n{ocr_text}"
                except Exception as e:
                    print(f"⚠️ [Warning] (data_loader.extract_text_from_pdf) 페이지 {page_num + 1} 처리 중 오류: {e}")
    except Exception as e:
        raise RuntimeError(f"❌ [Runtime] (data_loader.extract_text_from_pdf) PDF 파일 처리 오류: {e}")

    return full_text


def retrieve_top_documents_from_metadata(
    query, csv_path, embeddings, chat_history, top_k=5
):
    """
    사용자 질문과 메타데이터를 기반으로 유사도 검색을 수행합니다.

    Args:
        query (str): 사용자 검색 쿼리
        csv_path (str): CSV 메타데이터 파일 경로
        embeddings: 사전 로드된 임베딩 모델 인스턴스
        chat_history (str): 사용자 채팅 이력 문자열
        top_k (int): 반환할 상위 문서 수 (기본 5)

    Returns:
        pd.DataFrame: 유사도와 함께 반환된 상위 문서들의 메타데이터 DataFrame

    Raises:
        FileNotFoundError: 입력 파일 없음
        ValueError: CSV 로딩 실패 또는 필수 열 누락
        RuntimeError: 임베딩 텍스트 생성 또는 결과 계산 중 오류
    """
    if embeddings is None:
        raise ValueError("❌ [Value] (data_loader.retrieve_top_documents_from_metadata) embedder 인자 누락")

    if chat_history is None:
        raise ValueError("❌ [Value] (data_loader.retrieve_top_documents_from_metadata) chat_history 인자 누락")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"❌ [FileNotFound] (data_loader.retrieve_top_documents_from_metadata) 파일을 찾을 수 없습니다: {csv_path}"
        )

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(
            f"❌ [Value] (data_loader.retrieve_top_documents_from_metadata) CSV 파일 로딩 실패: {e}"
        )

    required_columns = ["사업명", "발주 기관", "사업 요약", "파일명"]
    for col in required_columns:
        if col not in df.columns:
            raise KeyError(
                f"❌ [Key] (data_loader.retrieve_top_documents_from_metadata) '{col}' 열이 CSV에 존재하지 않습니다."
            )

    def make_embedding_text(row):
        return f"{chat_history} {row['파일명']} {row['사업 요약']} {row['사업명']} {row['발주 기관']}"

    try:
        df["임베딩텍스트"] = df.apply(make_embedding_text, axis=1)
    except Exception as e:
        raise RuntimeError(f"❌ (data_loader.retrieve_top_documents_from_metadata) 임베딩 텍스트 생성 중 오류: {e}")

    doc_texts = df["임베딩텍스트"].tolist()

    if hasattr(embeddings, "encode"):
        doc_embeddings = embeddings.encode(doc_texts, convert_to_tensor=True)
        query_embedding = embeddings.encode(query, convert_to_tensor=True)
        similarities = cos_sim(query_embedding, doc_embeddings)[0].cpu().numpy()
    else:
        doc_embeddings = embeddings.embed_documents(doc_texts)
        query_embedding = embeddings.embed_query(query)
        similarities = cosine_similarity(
            np.array([query_embedding]), np.array(doc_embeddings)
        )[0]

    top_k_indices = np.argsort(similarities)[::-1][:top_k]

    try:
        top_docs = df.iloc[top_k_indices].copy()
        top_docs["유사도"] = similarities[top_k_indices]
    except Exception as e:
        raise RuntimeError(
            f"❌ [Runtime] (data_loader.retrieve_top_documents_from_metadata) 결과 DataFrame 생성 실패: {e}"
        )

    table = [[idx, row["파일명"], f"{row['유사도']:.4f}"] for idx, row in top_docs.iterrows()]
    output = tabulate(table, headers=["IDX", "파일명", "유사도"], tablefmt="github")
    print("\n".join("    " + line for line in output.splitlines()))

    return top_docs


def data_process(df: pd.DataFrame, config: dict, apply_ocr: bool = True, file_type: str = "all") -> pd.DataFrame:
    """
    주어진 파일 목록(DataFrame)을 기반으로 HWP 또는 PDF 파일을 읽어 텍스트를 추출합니다.
    추출된 텍스트는 'full_text' 컬럼에 저장되며, OCR을 사용할 수 있습니다.

    Args:
        df (pd.DataFrame): '파일명' 컬럼을 포함한 입력 데이터프레임
        config (dict): 프로젝트 설정 정보가 담긴 딕셔너리
        apply_ocr (bool): PDF 파일 처리 시 OCR(optical character recognition) 적용 여부
        file_type (str): 처리할 파일 유형 ('hwp', 'pdf', 'all')

    Returns:
        pd.DataFrame: 'full_text' 컬럼이 추가된 파일 처리 결과 데이터프레임

    Raises:
        FileNotFoundError: 특정 파일 경로가 존재하지 않을 경우
        RuntimeError: 파일 처리 중 오류가 발생한 경우
    """
    base_dir = config['settings']['project_root']
    file_root = os.path.join(base_dir, "data", "files")

    if file_type in ["hwp", "pdf"]:
        mask = df["파일명"].str.lower().str.endswith(f".{file_type}")
        filtered_df = df[mask].copy()
    elif file_type == "all":
        filtered_df = df.copy()

    filtered_df["full_text"] = None

    for file_name in filtered_df["파일명"]:
        file_path = os.path.join(file_root, file_name)
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(
                    f"❌ [FileNotFound] (data_loader.data_process.path) 파일이 존재하지 않습니다: {file_path}"
                )

            if file_name.lower().endswith(".hwp") and file_type in ["hwp", "all"]:
                loader = HWPLoader(file_path)
                docs = loader.load()
                if docs and isinstance(docs[0].page_content, str):
                    filtered_df.loc[filtered_df["파일명"] == file_name, "full_text"] = docs[0].page_content
                else:
                    print(f"⚠️ [Warning] (data_loader.data_process.hwp) HWP 파일 무시됨 (내용 없음): {file_name}")

            elif file_name.lower().endswith(".pdf") and file_type in ["pdf", "all"]:
                text = extract_text_from_pdf(Path(file_path), apply_ocr=apply_ocr)
                filtered_df.loc[filtered_df["파일명"] == file_name, "full_text"] = text

            else:
                print(f"⚠️ [Warning] (data_loader.data_process) 지원하지 않는 파일 형식입니다: {file_name}")

        except Exception as e:
            raise RuntimeError(
                f"❌ [Runtime] (data_loader.data_process) 파일 처리 오류 ({file_name}): {e}"
            )
    empty_files = filtered_df[filtered_df["full_text"].isna()]["파일명"].tolist()
    if empty_files:
        print(f"⚠️ [Warning] (data_loader.data_process) 다음 파일은 내용이 없습니다: {', '.join(empty_files)}")

    return filtered_df.reset_index(drop=True)