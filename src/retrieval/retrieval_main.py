import yaml
import os
import sys

src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_dir = os.path.dirname(os.path.abspath(src_dir))
                           
if src_dir not in sys.path:
    sys.path.append(src_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)
    
from retrieval import retrieve_documents
from vector_db import load_vector_db
from main import generate_index_name

def run_retrieval(vector_store=None, verbose=False):
    with open(os.path.join(root_dir, "config.yaml"), "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    index_name = generate_index_name(config)
    
    embed_config = config.get("embedding", {})
    vector_db_path = os.path.join(root_dir, embed_config.get("vector_db_path", ""))
    embed_model = embed_config.get("embed_model", "openai")
    db_type = embed_config.get("db_type", "faiss")

    if vector_store is None:
        vector_store=load_vector_db(vector_db_path, embed_model, index_name=index_name, db_type=db_type)
        if verbose:
            print("✅ Vector DB 로드 완료")
        
    docs = retrieve_documents(query=config.get("retriever", {}).get("query", ""), 
                                 vector_store=vector_store,
                                 top_k=8, 
                                 search_type="similarity", 
                                 all_chunks=None)
    if verbose:
        print(f"    -임베딩 모델: {embed_model}")
        print(f"    -DB 타입: {db_type}")
        print(f"    -벡터 DB 경로: {vector_db_path}")
        print(f"    -벡터 DB 파일: {index_name}")
        for i, doc in enumerate(docs, 1):
            print(f"\n📄 문서 {i}")
            print(f"본문:\n{doc['page_content']}...")
            print(f"메타데이터: {doc['metadata']}")
            
    return docs

run_retrieval(vector_store=None, verbose=True)