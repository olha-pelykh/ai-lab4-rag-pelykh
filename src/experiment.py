# src/experiment.py
# Е2: переранжування top-20 dense-пошуку cross-encoder'ом
import csv, time
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import CrossEncoder
from index import get_embeddings, load_store
from eval_retrieval import load_questions, hit_at_k, reciprocal_rank, K_MAX

RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
NAME, FETCH_K = "A", 20      # конфігурація, обрана за результатами оцінювання пошуку

if __name__ == "__main__":
    qs = [q for q in load_questions() if q["gold"]]
    store = load_store(NAME, get_embeddings())
    ce = CrossEncoder(RERANKER)
    store.similarity_search_with_score(qs[0]["question"], k=FETCH_K)      # прогрів
    ce.predict([(qs[0]["question"], "warm-up")])

    ranks = {"dense": {}, "dense+rerank": {}}
    t_dense, t_rerank = [], []
    for q in qs:
        t0 = time.perf_counter()
        hits = store.similarity_search_with_score(q["question"], k=FETCH_K)
        t1 = time.perf_counter()
        docs = [d for d, _ in hits]
        scores = ce.predict([(q["question"], d.page_content) for d in docs])
        order = np.argsort(scores)[::-1]
        t2 = time.perf_counter()
        ranks["dense"][q["qid"]] = [d.metadata["page"] for d in docs][:K_MAX]
        ranks["dense+rerank"][q["qid"]] = [docs[i].metadata["page"] for i in order][:K_MAX]
        t_dense.append(t1 - t0); t_rerank.append(t2 - t1)

    ks = range(1, K_MAX + 1)
    with open("eval/experiment_e2.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["system", "Hit@1", "Hit@3", "Hit@5", "MRR", "t_search_s"])
        for s, t in [("dense", np.mean(t_dense)),
                     ("dense+rerank", np.mean(t_dense) + np.mean(t_rerank))]:
            curve = [np.mean([hit_at_k(ranks[s][q["qid"]], q["gold"], k) for q in qs])
                     for k in ks]
            mrr = np.mean([reciprocal_rank(ranks[s][q["qid"]], q["gold"]) for q in qs])
            w.writerow([s, f"{curve[0]:.2f}", f"{curve[2]:.2f}", f"{curve[4]:.2f}",
                        f"{mrr:.2f}", f"{t:.3f}"])
            plt.plot(ks, curve, marker="o", label=s)
    plt.xlabel("k"); plt.ylabel("Hit@k"); plt.ylim(0, 1.05)
    plt.grid(alpha=0.3); plt.legend()
    plt.savefig("reports/figures/08_rerank_hit_at_k.png", dpi=150); plt.close()
    for q in qs:
        a = ranks["dense"][q["qid"]]; b = ranks["dense+rerank"][q["qid"]]
        ra = next((i for i, p in enumerate(a, 1) if p in q["gold"]), None)
        rb = next((i for i, p in enumerate(b, 1) if p in q["gold"]), None)
        print(q["qid"], "ранг dense:", ra, "ранг після rerank:", rb)
    print(f"час dense: {np.mean(t_dense):.3f} с, час rerank: {np.mean(t_rerank):.3f} с")
