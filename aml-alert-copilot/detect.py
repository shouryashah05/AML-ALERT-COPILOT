"""Detection: 3 typology rules (with evidence) + IsolationForest anomaly score -> ranked alerts."""
import numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

T = 10_000                      # cash reporting threshold (USD)
HIGH_RISK = {"IR", "KP", "MM"}  # illustrative list


def load(path="data/transactions.csv"):
    return pd.read_csv(path, parse_dates=["ts"]).sort_values("ts").reset_index(drop=True)


# ---------------- typology rules: each returns {account: hit} with evidence txn ids ----------------
def rule_structuring(tx):
    """>=3 cash deposits in [0.8T, T) within 7 days, making up >=60% of that window's cash deposits."""
    hits = {}
    cash_in = tx[(tx.channel == "cash") & (tx.direction == "in")]
    for acc, g in cash_in.groupby("account_id"):
        best = None
        for t0 in g.ts:
            w = g[(g.ts >= t0) & (g.ts < t0 + pd.Timedelta(days=7))]
            band = w[(w.amount >= 0.8 * T) & (w.amount < T)]
            if len(band) >= 3 and len(band) / len(w) >= 0.6 and (best is None or len(band) > len(best)):
                best = band
        if best is not None:
            hits[acc] = dict(rule="structuring", severity=min(1.0, 0.5 + 0.1 * len(best)),
                             txn_ids=best.txn_id.tolist(), count=len(best),
                             total_amount=round(best.amount.sum(), 2),
                             detail=f"{len(best)} cash deposits of ${best.amount.min():,.0f}-${best.amount.max():,.0f} "
                                    f"within 7 days, totalling ${best.amount.sum():,.0f}, just under the ${T:,} reporting threshold")
    return hits


def rule_pass_through(tx):
    """Inbound wire/transfer >= $20k with >=80% sent out within 72h (rapid movement of funds)."""
    hits = {}
    big_in = tx[(tx.direction == "in") & tx.channel.isin(["wire", "transfer"]) & (tx.amount >= 20_000)]
    for r in big_in.itertuples():
        out = tx[(tx.account_id == r.account_id) & (tx.direction == "out")
                 & (tx.ts > r.ts) & (tx.ts <= r.ts + pd.Timedelta(hours=72))]
        ratio = out.amount.sum() / r.amount
        if ratio >= 0.8 and r.account_id not in hits:
            hits[r.account_id] = dict(rule="pass_through", severity=min(1.0, 0.5 + 0.5 * ratio),
                                      txn_ids=[r.txn_id] + out.txn_id.tolist(), count=len(out) + 1,
                                      total_amount=round(r.amount, 2),
                                      detail=f"inbound ${r.amount:,.0f} followed by ${out.amount.sum():,.0f} "
                                             f"({ratio:.0%}) sent out within 72h to {out.counterparty_id.nunique()} counterparties")
    return hits


def rule_high_risk_geo(tx):
    """>=2 txns with high-risk jurisdictions totalling >= $10k."""
    hits = {}
    for acc, g in tx[tx.counterparty_country.isin(HIGH_RISK)].groupby("account_id"):
        if len(g) >= 2 and g.amount.sum() >= 10_000:
            hits[acc] = dict(rule="high_risk_geo", severity=min(1.0, 0.5 + g.amount.sum() / 100_000),
                             txn_ids=g.txn_id.tolist(), count=len(g), total_amount=round(g.amount.sum(), 2),
                             detail=f"{len(g)} transactions totalling ${g.amount.sum():,.0f} with high-risk jurisdictions "
                                    f"({', '.join(sorted(g.counterparty_country.unique()))})")
    return hits


# ---------------- unsupervised anomaly score + per-feature explanation ----------------
def features(tx):
    g = tx.groupby("account_id")
    f = pd.DataFrame({"txn_count": g.size(), "mean_amt": g.amount.mean(), "max_amt": g.amount.max(),
                      "std_amt": g.amount.std(), "n_counterparties": g.counterparty_id.nunique()})
    vol = g.amount.sum()
    def vol_of(mask): return tx[mask].groupby("account_id").amount.sum().reindex(f.index).fillna(0)
    f["out_in_ratio"] = vol_of(tx.direction == "out") / vol_of(tx.direction == "in").clip(lower=1)
    f["cash_share"] = vol_of(tx.channel == "cash") / vol
    f["wire_share"] = vol_of(tx.channel == "wire") / vol
    f["high_risk_share"] = vol_of(tx.counterparty_country.isin(HIGH_RISK)) / vol
    near = (tx.channel == "cash") & (tx.direction == "in") & (tx.amount >= 0.8 * T) & (tx.amount < T)
    f["near_threshold_cash"] = tx[near].groupby("account_id").size().reindex(f.index).fillna(0)
    daily = tx.assign(day=tx.ts.dt.floor("D")).groupby(["account_id", "day"]).amount.sum().unstack(fill_value=0)
    f["peak_7d_share"] = daily.T.rolling(7).sum().max() / vol
    return f.fillna(0).clip(lower=0)


def anomaly(f):
    L = np.log1p(f)
    iso = IsolationForest(n_estimators=300, random_state=0).fit(StandardScaler().fit_transform(L))
    raw = -iso.score_samples(StandardScaler().fit_transform(L))
    pct = pd.Series(raw, index=f.index).rank(pct=True)          # 0..1, 1 = most anomalous
    med = L.median(); scale = (1.4826 * (L - med).abs().median()).where(lambda s: s > 0, L.std()).clip(lower=1e-6)
    z = (L - med) / scale                                         # robust z-score vs peers
    expl = {a: [dict(feature=k, value=round(float(f.loc[a, k]), 3), peer_median=round(float(np.expm1(med[k])), 3),
                     robust_z=round(float(v), 1)) for k, v in z.loc[a].nlargest(3).items()] for a in f.index}
    return pct, expl


# ---------------- combine ----------------
def run(tx):
    f = features(tx)
    pct, expl = anomaly(f)
    hits = {}
    for rule in (rule_structuring, rule_pass_through, rule_high_risk_geo):
        for a, h in rule(tx).items():
            hits.setdefault(a, []).append(h)
    s = pd.DataFrame({"anomaly_pct": pct.round(3)})
    s["rule_sev"] = [max([h["severity"] for h in hits.get(a, [])], default=0.0) for a in s.index]
    s["risk"] = (0.65 * s.rule_sev + 0.35 * s.anomaly_pct).round(3)
    s["rules"] = [",".join(h["rule"] for h in hits.get(a, [])) for a in s.index]
    s["alert"] = (s.rule_sev > 0) | (s.anomaly_pct >= 0.97)
    return s.sort_values("risk", ascending=False), hits, expl


if __name__ == "__main__":
    scores, hits, _ = run(load())
    print(scores[scores.alert].head(15))
