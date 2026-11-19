# src/chunking.py
import json
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import EMB_MODEL, CHUNKS

def make_chunks(pages, size, overlap, tok):
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tok, chunk_size=size, chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""])
    docs = [Document(page_content=p["text"], metadata={"page": p["page"]})
            for p in pages]
    chunks = splitter.split_documents(docs)
    return [{"chunk_id": i, "page": c.metadata["page"], "text": c.page_content}
            for i, c in enumerate(chunks)]

if __name__ == "__main__":
    pages = json.load(open("data/pages.json", encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(EMB_MODEL)
    for name, (size, overlap) in CHUNKS.items():
        chunks = make_chunks(pages, size, overlap, tok)
        lens = [len(tok.encode(c["text"], add_special_tokens=False))
                for c in chunks]
        print(f"{name}: size={size}, overlap={overlap}, чанків={len(chunks)}, "
              f"середня довжина={sum(lens) / len(lens):.0f} ток.")
        with open(f"data/chunks_{name}.json", "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False)
        plt.hist(lens, bins=40, alpha=0.6, label=f"{name}: {size}/{overlap}")
    plt.xlabel("Довжина чанка, токенів"); plt.ylabel("Кількість чанків")
    plt.legend()
    plt.savefig("reports/figures/02_chunk_lengths.png", dpi=150)
