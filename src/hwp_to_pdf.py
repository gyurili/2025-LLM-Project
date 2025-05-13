import os
import win32com.client as win32

def hwp_to_pdf(hwp_path, pdf_path):
    hwp = win32.gencache.EnsureDispatch("HWPFrame.HwpObject")
    hwp.XHwpWindows.Item(0).Visible = True

    # 파일 열기
    hwp.Open(hwp_path)

    # PDF로 저장
    hwp.SaveAs(pdf_path, "PDF")

    # 종료
    hwp.Quit()

def batch_convert_hwp_to_pdf(folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".hwp"):
            hwp_path = os.path.join(folder_path, filename)
            pdf_filename = os.path.splitext(filename)[0] + ".pdf"
            pdf_path = os.path.join(folder_path, pdf_filename)

            try:
                hwp_to_pdf(hwp_path, pdf_path)
                print(f"변환 완료: {pdf_filename}")
            except Exception as e:
                print(f"오류 발생: {filename} → {e}")

if __name__ == "__main__":
    # 사용 예시
    folder_path = r"C:\Users\user\Desktop\PythonWorkspace\2025-LLM-Project\data\files"
    batch_convert_hwp_to_pdf(folder_path)