# 2025-LLM-Project 리드미

 ## streamlit 실행코드:

 - 실행 코드: 
 
     streamlit run app.py --server.address=0.0.0.0 --server.port=8501

   **Note:** 최대 문서 수를 100으로 지정한다면, 처음 한번 전체 vector_db를 생성하고 이후로는 load_vector_db로 진행됨<br>
    -> pros: 처음 5분간의 소요 시간이후 다음 query 부터는 1초 내외의 소요시간 발생<br>
    -> 최대 문서 수를 100 미만으로 지정한다면(예: config['data']['top_k'] == 5의 경우 매번 vector_db를 질문과 맞게 생성하게 되며 소요시간은 각각 30초내외이다.)
