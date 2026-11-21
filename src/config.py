# src/config.py
# Параметри варіанта 12 (таблиця 6.1): Microbiology (OpenStax)
VARIANT = 12
DOC_PATH = "data/document.pdf"
EMB_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX, PASSAGE_PREFIX = "", ""      # для моделей e5: "query: ", "passage: "
CHUNKS = {"A": (256, 32), "B": (512, 64)}   # (chunk_size, chunk_overlap) у токенах
LLM_MODEL = "qwen2.5:3b"                    # модель Ollama
TOP_K = 3                                   # кількість чанків у контексті (плато Hit@k у конфігурації A)
THRESHOLD = 0.74                            # поріг відмови за косинусною схожістю top-1 (конфігурація A)
SEED = VARIANT
