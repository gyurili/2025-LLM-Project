import os

def set_cache_dirs():
    os.environ['TRANSFORMERS_CACHE'] = os.path.expanduser('~/.cache/huggingface/transformers')
    os.environ['SENTENCE_TRANSFORMERS_HOME'] = os.path.expanduser('~/.cache/sbert')
    os.environ['XDG_CACHE_HOME'] = os.path.expanduser('~/.cache/xdg')
    os.environ['LANGCHAIN_CACHE'] = os.path.expanduser('~/.cache/langchain')
    os.environ['TORCH_HOME'] = os.path.expanduser('~/.cache/torch')