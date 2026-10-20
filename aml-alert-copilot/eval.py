"""Evaluation vs ground truth. NOTE: data is SYNTHETIC - this validates the pipeline, not real-world accuracy.
Run: python eval.py"""
import numpy as np, pandas as pd
import detect, copilot

tx = detect.load()
accounts = pd.read_csv("data/accounts.csv").set_index("account_id")
labels = pd.read_csv("data/labels.csv").set_index("account_id")
scores, hits, expl = detect.run(tx)
y = labels.loc[scores.index, "suspicious"]


def report(name, flagged):
    pred = pd.Series(scores.index.isin(flagged), index=scores.index)
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    print(f"{name:<38} alerts={tp + fp:>3}  precision={p:.2f}  recall={r:.2f}  F1={2 * p * r / max(p + r, 1e-9):.2f}")


print("== Detection (account level) ==")
report("A. static rule: any txn >= $10k", tx[tx.amount >= 10_000].account_id.unique())
report("B. anomaly model only (top 5%)", scores.index[scores.anomaly_pct >= 0.95])
report("C. rules only", scores.index[scores.rule_sev > 0])
report("D. hybrid (rules + anomaly)  <- ours", scores.index[scores.alert])

k = int(y.sum()); topk = scores.index[:k]
print(f"\nprecision@{k} by risk score: {y.loc[topk].mean():.2f}")
alerts = scores[scores.alert]
print("\nRecall by typology (hybrid):")
print(labels.loc[labels.suspicious == 1].assign(hit=lambda d: d.index.isin(alerts.index)).groupby("typology").hit.agg(["sum", "count"]).rename(columns={"sum": "caught", "count": "total"}))
print("\nFalse positives by account type:", labels.loc[alerts.index].query("suspicious == 0").typology.value_counts().to_dict())
print("Missed:", [f"{a} ({labels.typology[a]})" for a in labels.index[(labels.suspicious == 1) & ~labels.index.isin(alerts.index)]])

print("\n== Copilot grounding ==")
ok = caught = 0
for a in alerts.index:
    c = copilot.build_case(a, scores, hits, expl, tx, accounts)
    n = copilot.template_narrative(c)
    ok += not copilot.validate(n, c)
    bad = dict(n, cited_txn_ids=n["cited_txn_ids"] + ["T999999"], summary=n["summary"] + " Total of $123,456 moved.")
    caught += bool(copilot.validate(bad, c))
print(f"template narratives passing validator : {ok}/{len(alerts)}")
print(f"injected hallucinations caught        : {caught}/{len(alerts)}")
