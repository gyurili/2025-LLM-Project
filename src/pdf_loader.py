import fitz  # PyMuPDF - PDF 열기 및 페이지 렌더링용
import easyocr  # OCR 엔진
import numpy as np
import pandas as pd
from PIL import Image  # 이미지 처리
from pathlib import Path
from tqdm import tqdm  # 진행률 표시

# EasyOCR Reader 객체 생성 (GPU 사용)
# 한글(ko) + 영어(en)를 인식하며, 모델은 한 번만 로드됨
reader = easyocr.Reader(['ko', 'en'], gpu=True)

# OCR 안전 수행 함수
def safe_ocr(img_array: np.ndarray, ocr_reader: easyocr.Reader) -> str:
    """
    이미지 배열을 입력받아 EasyOCR로 텍스트를 추출합니다.

    매개변수:
        img_array (np.ndarray): OCR을 수행할 이미지 배열
        ocr_reader (easyocr.Reader): 초기화된 EasyOCR 리더 인스턴스

    반환값:
        str: 추출된 텍스트 문자열 (오류 발생 시 공백 또는 오류 메시지)
    """
    try:
        result = ocr_reader.readtext(img_array, detail=0)
        if not isinstance(result, list):
            return ""
        return "\n".join(result)
    except Exception as e:
        return f"[OCR 실패: {e}]"


# 단일 PDF 파일에서 텍스트 + OCR 텍스트 추출
def extract_text_from_pdf_with_ocr(pdf_path: Path, apply_ocr: bool = True) -> str:
    """
    한 개의 PDF 파일에서 일반 텍스트를 추출하고,
    필요시 모든 페이지에 대해 OCR을 수행한 결과를 결합합니다.

    매개변수:
        pdf_path (Path): 처리할 PDF 파일 경로
        apply_ocr (bool): OCR 수행 여부 (기본값: True)

    반환값:
        str: 해당 PDF 전체에서 추출한 텍스트 (OCR 결과 포함 가능)
    """
    full_text = ""

    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(tqdm(doc, desc=f"{pdf_path.name}")):
            # 1. 일반 텍스트 추출
            page_text = page.get_text()
            full_text += page_text

            # 2. OCR 조건 수행
            if apply_ocr:
                pix = page.get_pixmap(dpi=300)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                ocr_text = safe_ocr(np.array(img), reader)

                if ocr_text.strip():
                    full_text += f"\n[OCR p.{page_num + 1}]\n{ocr_text}"

    return full_text


# 폴더 내 모든 PDF 파일을 일괄 처리
def process_all_pdfs_in_folder(folder_path: str, apply_ocr: bool = True) -> pd.DataFrame:
    """
    지정된 폴더 내 모든 PDF 파일을 대상으로 텍스트 및 OCR 추출을 수행합니다.

    매개변수:
        folder_path (str): 처리할 PDF 폴더 경로
        apply_ocr (bool): OCR 수행 여부 (기본값: True)

    반환값:
        pd.DataFrame: 각 파일명과 해당 텍스트를 담은 데이터프레임
    """
    pdf_files = list(Path(folder_path).glob("*.pdf"))
    results = []

    for idx, file_path in enumerate(pdf_files, start=1):
        print(f"[INFO] 파일 {idx}/{len(pdf_files)} 처리 중: {file_path.name}")
        try:
            text = extract_text_from_pdf_with_ocr(file_path, apply_ocr=apply_ocr)
        except Exception as e:
            print(f"[ERROR] {file_path.name}: {e}")
            continue

        results.append({
            "파일명": file_path.name,
            "내용": text
        })

    return pd.DataFrame(results)