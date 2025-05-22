import os

def get_project_root_dir(project_name: str = "2025-LLM-Project") -> str:
    """
    현재 파일 또는 실행 위치에서 프로젝트 루트 디렉토리를 탐색합니다.
    
    Args:
        project_name (str): 루트 디렉토리명
    
    Returns:
        str: 루트 디렉토리의 절대 경로
    """
    cwd = os.path.abspath(os.getcwd())
    while cwd != os.path.dirname(cwd):  # 루트까지 올라감
        if os.path.basename(cwd) == project_name:
            return cwd
        cwd = os.path.dirname(cwd)
    raise RuntimeError(f"❌ 루트 디렉토리({project_name})를 찾을 수 없습니다.")