"""Synthetic AML data: normal behaviour + 3 injected typologies + benign look-alikes.
Run: python generate_data.py  ->  data/transactions.csv, data/accounts.csv, data/labels.csv
"""
import numpy as np, pandas as pd

rng = np.random.default_rng(42)
N_ACC, DAYS = 600, 60
START = pd.Timestamp("2026-07-01")
COUNTRIES = ["IN", "US", "GB", "AE", "SG"]
HIGH_RISK = ["IR", "KP", "MM"]  # illustrative; use an official FATF list in real use
T = 10_000                      # cash reporting threshold (USD)

acc = pd.DataFrame({
    "account_id": [f"ACC{i:04d}" for i in range(N_ACC)],
    "segment": rng.choice(["retail", "business"], N_ACC, p=[0.8, 0.2]),
    "home_country": rng.choice(COUNTRIES, N_ACC),
})
biz = acc.segment == "business"
acc["typical_amt"] = np.where(biz, rng.lognormal(7.0, 0.5, N_ACC), rng.lognormal(5.5, 0.6, N_ACC))
acc["monthly_txns"] = np.where(biz, rng.integers(25, 60, N_ACC), rng.integers(8, 25, N_ACC))
home = dict(zip(acc.account_id, acc.home_country))

rows = []
def add(a, ts, direction, channel, amt, ctry=None, cp=None):
    cp = cp or (f"BR{rng.integers(0, 40):02d}" if channel == "cash" else f"CP{rng.integers(0, 3000):04d}")
    rows.append(dict(account_id=a, ts=ts, direction=direction, channel=channel,
                     amount=round(float(amt), 2), counterparty_id=cp,
                     counterparty_country=ctry or home[a]))

def day(d, extra_hours=0.0):
    return START + pd.Timedelta(days=float(d)) + pd.Timedelta(hours=float(extra_hours))

# ---- 1. normal behaviour for everyone ----
for r in acc.itertuples():
    for _ in range(rng.poisson(r.monthly_txns * DAYS / 30)):
        ch = rng.choice(["card", "transfer", "cash", "wire"], p=[0.5, 0.3, 0.12, 0.08])
        p_in = {"card": 0.0, "transfer": 0.45, "cash": 0.6, "wire": 0.45}[ch]
        amt = r.typical_amt * rng.lognormal(0, 0.6) * (3 if ch == "wire" else 1)
        if ch == "cash":
            amt = min(amt, 7_500)
        ctry = rng.choice(COUNTRIES) if ch in ("wire", "transfer") and rng.random() < 0.3 else None
        add(r.account_id, day(rng.uniform(0, DAYS)), "in" if rng.random() < p_in else "out", ch, amt, ctry)

# ---- 2. inject typologies on disjoint accounts ----
pool = list(rng.permutation(acc.account_id))
def take(n): return [pool.pop() for _ in range(n)]
labels = {a: "none" for a in acc.account_id}

# structuring: 4-7 cash deposits just under the threshold within ~4 days
for a in take(10):
    labels[a] = "structuring"; d0 = rng.uniform(5, 50)
    for _ in range(rng.integers(4, 8)):
        add(a, day(d0 + rng.uniform(0, 4)), "in", "cash", rng.uniform(8_500, 9_900))

# pass-through: big wire in, ~90-98% sent out within 48h to several counterparties
for a in take(10):
    labels[a] = "pass_through"; d0 = rng.uniform(5, 55); amt = rng.uniform(30_000, 120_000)
    add(a, day(d0), "in", "wire", amt, rng.choice(COUNTRIES + HIGH_RISK[:1]))
    k = rng.integers(2, 5); parts = rng.dirichlet(np.ones(k)) * amt * rng.uniform(0.88, 0.98)
    for p in parts:
        add(a, day(d0, rng.uniform(1, 48)), "out", rng.choice(["wire", "transfer"]), p, rng.choice(["AE", "SG", "HK"]))

# high-risk geography: burst of wires with FATF-style high-risk countries
for a in take(8):
    labels[a] = "high_risk_geo"; d0 = rng.uniform(5, 45)
    for _ in range(rng.integers(3, 7)):
        add(a, day(d0 + rng.uniform(0, 10)), rng.choice(["in", "out"]), "wire",
            rng.uniform(3_000, 9_000), rng.choice(HIGH_RISK))

# ---- 3. benign look-alikes (hard negatives) ----
# (a) cash-intensive businesses: heavy legit cash, a few CTR-sized deposits
for a in take(8):
    labels[a] = "benign_cash_business"
    for d in range(DAYS):
        if rng.random() < 0.85:
            add(a, day(d, rng.uniform(9, 18)), "in", "cash", np.clip(rng.normal(5_500, 1_800), 800, 9_800))
    for _ in range(rng.integers(2, 5)):
        add(a, day(rng.uniform(0, DAYS)), "in", "cash", rng.uniform(10_500, 18_000))
# (b) one large wire in (property sale / loan), kept in the account
for a in take(6):
    labels[a] = "benign_large_wire"
    add(a, day(rng.uniform(5, 40)), "in", "wire", rng.uniform(60_000, 150_000))
# (c) family remitter: small, steady transfers to a high-risk country
for a in take(3):
    labels[a] = "benign_remitter"
    for d in (10, 40):
        add(a, day(d + rng.uniform(0, 3)), "out", "transfer", rng.uniform(200, 400), rng.choice(HIGH_RISK))

tx = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
tx.insert(0, "txn_id", [f"T{i:06d}" for i in range(len(tx))])
lab = pd.DataFrame({"account_id": list(labels), "typology": list(labels.values())})
lab["suspicious"] = lab.typology.isin(["structuring", "pass_through", "high_risk_geo"]).astype(int)

tx.to_csv("data/transactions.csv", index=False)
acc[["account_id", "segment", "home_country"]].to_csv("data/accounts.csv", index=False)
lab.to_csv("data/labels.csv", index=False)
print(f"{len(tx):,} txns | {N_ACC} accounts | {lab.suspicious.sum()} suspicious | {lab.typology.value_counts().to_dict()}")
