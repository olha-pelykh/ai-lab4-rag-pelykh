# src/rag.py
import csv, time
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from index import get_embeddings, load_store
from eval_retrieval import load_questions
from config import LLM_MODEL, TOP_K, THRESHOLD, SEED

REFUSAL = "НЕМАЄ У ДОКУМЕНТІ"
SYSTEM = ("You are an assistant that answers in English ONLY on the basis of the "
          "provided context. After each statement give the source in the format "
          "[стор. N]. Answer in one or two sentences. Only if the context has no "
          f"information relevant to the question, write exactly: {REFUSAL}.")
rag_prompt = ChatPromptTemplate.from_messages(
    [("system", SYSTEM), ("human", "Контекст:\n{context}\n\nПитання: {question}")])
plain_prompt = ChatPromptTemplate.from_messages([("human", "{question}")])

llm = ChatOllama(model=LLM_MODEL, temperature=0, seed=SEED)
rag_chain = rag_prompt | llm | StrOutputParser()      # LCEL: prompt -> LLM -> текст
plain_chain = plain_prompt | llm | StrOutputParser()  # базова лінія без RAG

def format_docs(docs):
    return "\n\n".join(f"[стор. {d.metadata['page']}] {d.page_content}"
                       for d in docs)

def ask(question, store, k=TOP_K, threshold=THRESHOLD):
    t0 = time.perf_counter()
    hits = store.similarity_search_with_score(question, k=k)
    t1 = time.perf_counter()
    if not hits or hits[0][1] < threshold:   # нічого схожого не знайдено
        return REFUSAL, [], t1 - t0, 0.0
    docs = [d for d, _ in hits]
    text = rag_chain.invoke({"context": format_docs(docs), "question": question})
    pages = [d.metadata["page"] for d in docs]
    return text, pages, t1 - t0, time.perf_counter() - t1

if __name__ == "__main__":
    store = load_store("A", get_embeddings())      # краща конфігурація за результатами оцінювання пошуку
    with open("eval/answers.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["qid", "mode", "answer", "pages", "t_retr", "t_gen",
                    "label", "citation_ok"])
        for q in load_questions():
            text, pages, t_r, t_g = ask(q["question"], store)
            w.writerow([q["qid"], "rag", text, ";".join(map(str, pages)),
                        f"{t_r:.3f}", f"{t_g:.3f}", "", ""])
            t0 = time.perf_counter()
            text = plain_chain.invoke({"question": q["question"]})
            t_gen = time.perf_counter() - t0
            w.writerow([q["qid"], "llm", text, "", "0", f"{t_gen:.3f}", "", ""])
            f.flush()
            print(q["qid"], "готово", flush=True)
