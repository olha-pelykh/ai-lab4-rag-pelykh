# src/ingest.py
import json, re
import matplotlib.pyplot as plt
from pypdf import PdfReader
from transformers import AutoTokenizer
from config import DOC_PATH, EMB_MODEL

def load_pages(path):
    pages = []
    for i, page in enumerate(PdfReader(path).pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if text:
            pages.append({"page": i, "text": text})
    return pages

if __name__ == "__main__":
    pages = load_pages(DOC_PATH)
    tok = AutoTokenizer.from_pretrained(EMB_MODEL)
    n_tok = [len(tok.encode(p["text"], add_special_tokens=False)) for p in pages]
    print(f"Сторінок з текстом: {len(pages)}, токенів усього: {sum(n_tok)}, "
          f"медіана: {sorted(n_tok)[len(n_tok) // 2]}, максимум: {max(n_tok)}")
    with open("data/pages.json", "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False)
    plt.hist(n_tok, bins=40)
    plt.xlabel("Токенів на сторінці"); plt.ylabel("Кількість сторінок")
    plt.savefig("reports/figures/01_tokens_per_page.png", dpi=150)
