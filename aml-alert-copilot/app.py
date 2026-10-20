"""Analyst UI. Run: streamlit run app.py"""
import os, pandas as pd, streamlit as st
from datetime import datetime
import detect, copilot

st.set_page_config(page_title="AML Alert Copilot", layout="wide")

@st.cache_data
def load():
    tx = detect.load(); accounts = pd.read_csv("data/accounts.csv").set_index("account_id")
    scores, hits, expl = detect.run(tx)
    return tx, accounts, scores, hits, expl

tx, accounts, scores, hits, expl = load()
alerts = scores[scores.alert]

st.title("AML Alert Copilot")
st.caption("Synthetic data · decision support only - a human analyst makes every disposition")
m = st.columns(3)
m[0].metric("Accounts monitored", len(scores)); m[1].metric("Open alerts", len(alerts))
m[2].metric("Narrative mode", "Claude" if os.getenv("ANTHROPIC_API_KEY") else "Offline template")

left, right = st.columns([1, 2])
with left:
    st.subheader("Alert queue")
    st.dataframe(alerts[["risk", "rules", "anomaly_pct"]], height=480)
    acc = st.selectbox("Open case", alerts.index)

with right:
    case = copilot.build_case(acc, scores, hits, expl, tx, accounts)
    st.subheader(f"Case {acc}")
    k = st.columns(4)
    k[0].metric("Risk", f"{case['risk_score']:.2f}"); k[1].metric("Anomaly pctile", f"{case['anomaly_percentile']:.2f}")
    k[2].metric("Txns", case["stats"]["txn_count"]); k[3].metric("Largest txn", f"${case['stats']['largest_txn']:,.0f}")

    st.markdown("**Rule hits**")
    for h in case["rule_hits"]:
        st.warning(f"**{h['rule']}** - {h['detail']}")
    if not case["rule_hits"]:
        st.info("No typology rule fired - anomaly-only alert")

    st.markdown("**Why it deviates from peers**")
    st.dataframe(pd.DataFrame(case["top_deviations"]), hide_index=True)
    st.markdown("**Evidence transactions**")
    st.dataframe(pd.DataFrame(case["evidence_txns"]), hide_index=True)
    t = tx[tx.account_id == acc]
    st.bar_chart(t.assign(day=t.ts.dt.floor("D")).pivot_table(index="day", columns="direction", values="amount", aggfunc="sum", fill_value=0))

    st.markdown("**Copilot narrative**")
    key = f"narr_{acc}"
    if st.button("Generate narrative"):
        st.session_state[key] = copilot.narrative(case)
    if key in st.session_state:
        n, issues = st.session_state[key]
        st.write(n["summary"])
        st.write("Red flags:"); [st.write(f"- {f}") for f in n["red_flags"]]
        st.write(f"Suggested action: `{n['recommended_action']}` · cited: {', '.join(n['cited_txn_ids'])} · source: {n['source']}")
        if issues: st.error("Grounding check: " + "; ".join(issues))
        else: st.success("Grounding check passed - every cited txn and $ figure traced to case data")

    st.markdown("**Analyst decision**")
    with st.form(f"decision_{acc}"):
        d = st.radio("Disposition", copilot.ACTIONS, horizontal=True)
        note = st.text_input("Rationale")
        if st.form_submit_button("Save decision"):
            row = pd.DataFrame([dict(ts=datetime.now().isoformat(timespec="seconds"), account_id=acc,
                                     risk=case["risk_score"], decision=d, rationale=note)])
            row.to_csv("data/decisions.csv", mode="a", header=not os.path.exists("data/decisions.csv"), index=False)
            st.success("Saved to audit log")

if os.path.exists("data/decisions.csv"):
    st.subheader("Decision audit log")
    st.dataframe(pd.read_csv("data/decisions.csv").tail(10), hide_index=True)
