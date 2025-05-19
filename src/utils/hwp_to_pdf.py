import os
import platform
import subprocess

def hwp_to_pdf(hwp_path: str, pdf_path: str) -> None:
    """
    HWP 파일을 PDF 파일로 변환합니다.
    Windows에서는 win32com을 이용하고,
    macOS/Linux에서는 hwp5txt로 텍스트 추출 후 PDF로 저장합니다.

    Args:
        hwp_path (str): 변환할 .hwp 파일의 전체 경로
        pdf_path (str): 저장할 .pdf 파일의 전체 경로

    Returns:
        None

    Raises:
        FileNotFoundError: 입력 파일이 존재하지 않을 경우
        RuntimeError: 파일 열기나 저장 실패 시
    """
    if not os.path.exists(hwp_path):
        raise FileNotFoundError(f"HWP 파일이 존재하지 않습니다: {hwp_path}")
    
    system_type = platform.system()

    if system_type == "Windows":
        try:
            import win32com.client as win32
            hwp = win32.gencache.EnsureDispatch("HWPFrame.HwpObject")
            hwp.XHwpWindows.Item(0).Visible = True
        except Exception as e:
            raise RuntimeError("한글 프로그램을 불러오지 못했습니다. 한글과컴퓨터가 설치되어 있는지 확인하세요.") from e

        # 파일 열기
        try:
            hwp.Open(hwp_path)
        except Exception as e:
            raise RuntimeError(f"HWP 파일을 열 수 없습니다: {hwp_path}\n파일이 손상되었거나 HWP 형식이 아닌 파일일 수 있습니다.") from e

        # PDF로 저장
        try:
            hwp.SaveAs(pdf_path, "PDF")
        except Exception as e:
            raise RuntimeError(f"PDF로 저장하는 데 실패했습니다: {pdf_path}") from e
        # 한글 종료
        finally:
            hwp.Quit()
            
    elif system_type in ["Darwin", "Linux"]:
        # hwp5txt를 사용하여 HWP 파일을 텍스트로 변환
        try:
            txt_path = os.path.splitext(pdf_path)[0] + ".txt"
            subprocess.run(["hwp5txt", hwp_path, "-o", txt_path], check=True)

            from fpdf import FPDF
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            with open(txt_path, 'r', encoding='utf-8') as f:
                for line in f:
                    pdf.multi_cell(0, 10, line)
            pdf.output(pdf_path)
            os.remove(txt_path)

        except Exception as e:
            raise RuntimeError(f"텍스트 추출 또는 PDF 저장에 실패했습니다: {pdf_path}") from e

    else:
        raise RuntimeError(f"지원되지 않는 운영체제입니다: {system_type}")


def batch_convert_hwp_to_pdf(folder_path: str) -> None:
    """
    지정된 폴더 내의 모든 .hwp 파일을 찾아 .pdf 파일로 일괄 변환합니다.

    Args:
        folder_path (str): .hwp 파일들이 존재하는 폴더의 전체 경로

    Returns:
        None
    """
    if not os.path.exists(folder_path):
        print(f"지정된 폴더가 존재하지 않습니다: {folder_path}")
        return
    if not os.path.isdir(folder_path):
        print(f"지정된 경로가 폴더가 아닙니다: {folder_path}")
        return
    
    count = 0
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".hwp"):
            hwp_path = os.path.join(folder_path, filename)
            pdf_filename = os.path.splitext(filename)[0] + ".pdf"
            pdf_path = os.path.join(folder_path, pdf_filename)

            try:
                hwp_to_pdf(hwp_path, pdf_path)
                print(f"변환 완료: {pdf_filename}")
                count += 1
            except Exception as e:
                print(f"오류 발생: {filename} → {e}")

    if count == 0:
        print("변환할 HWP 파일이 없습니다.")
    else:
        print(f"총 {count}개의 HWP 파일이 PDF로 변환되었습니다.")


'''
if __name__ == "__main__":
    """
    스크립트 실행 시 동작:
    지정된 폴더 내 .hwp 파일을 모두 찾아 PDF로 변환합니다.
    """
    folder_path = r"C:\Users\user\Desktop\PythonWorkspace\2025-LLM-Project\data\files"
    batch_convert_hwp_to_pdf(folder_path)
'''