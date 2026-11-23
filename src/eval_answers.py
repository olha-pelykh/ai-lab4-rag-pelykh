# src/eval_answers.py
import pandas as pd
import matplotlib.pyplot as plt

LABELS = ["correct", "partial", "wrong", "hallucination",
          "refusal_ok", "refusal_wrong"]

df = pd.read_csv("eval/answers_labeled.csv")
q = pd.read_csv("eval/questions.csv", dtype=str).fillna("")
df = df.merge(q[["qid", "pages"]].rename(columns={"pages": "gold"}), on="qid")
assert df["label"].isin(LABELS).all(), "Розмітьте вручну всі відповіді"

for mode, g in df.groupby("mode"):
    ans, unans = g[g["gold"] != ""], g[g["gold"] == ""]
    correct = ((ans["label"] == "correct")
               + 0.5 * (ans["label"] == "partial")).mean()
    halluc = (g["label"] == "hallucination").mean()
    refusal = (unans["label"] == "refusal_ok").mean()
    print(f"{mode}: правильність={correct:.2f}, галюцинації={halluc:.2f}, "
          f"коректні відмови={refusal:.2f}")
rag = df[(df["mode"] == "rag") & df["citation_ok"].notna()]
print(f"Точність посилань на сторінки (rag): {rag['citation_ok'].mean():.2f}")

tab = pd.crosstab(df["mode"], df["label"], normalize="index")
tab = tab.reindex(columns=LABELS, fill_value=0)
tab.plot.barh(stacked=True, figsize=(8, 3))
plt.xlabel("Частка відповідей"); plt.legend(ncol=3, fontsize=8, loc="upper center",
                                            bbox_to_anchor=(0.5, -0.25))
plt.tight_layout(); plt.savefig("reports/figures/05_answer_labels.png", dpi=150)
plt.close()

r = df[df["mode"] == "rag"]
plt.boxplot([r["t_retr"], r["t_gen"]], tick_labels=["пошук", "генерація"])
plt.ylabel("Час, с"); plt.yscale("log")
plt.savefig("reports/figures/06_latency.png", dpi=150)
