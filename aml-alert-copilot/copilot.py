"""GenAI copilot: case packet -> grounded narrative (Claude, or offline template) -> validator.
The LLM only drafts. Every cited txn id and dollar figure is checked against the case packet."""
import os, re, json

ACTIONS = ["escalate_for_SAR_review", "monitor", "close_as_false_positive"]
MODEL = os.getenv("COPILOT_MODEL", "claude-sonnet-5-5")

SYSTEM = f"""You are an AML investigation copilot. Draft a case summary for a human investigator from the JSON case packet.
Rules:
- Use ONLY facts and numbers present in the packet. Copy dollar figures exactly as written; do no arithmetic or rounding.
- Cite transaction ids only from evidence_txns.
- Describe patterns as 'consistent with' - never state intent or guilt. A human makes the final decision.
- recommended_action must be one of {ACTIONS}.
Return ONE JSON object, no prose, with keys: summary (<=80 words), typology, red_flags (list of short strings),
recommended_action, cited_txn_ids (list)."""


def build_case(acc, scores, hits, expl, tx, accounts):
    t = tx[tx.account_id == acc]
    ids = sorted({i for h in hits.get(acc, []) for i in h["txn_ids"]})
    ev = t[t.txn_id.isin(ids)] if ids else t.nlargest(8, "amount")
    ev = ev.sort_values("ts").head(15)
    return {
        "account_id": acc,
        "profile": accounts.loc[acc].to_dict(),
        "period": f"{t.ts.min():%Y-%m-%d} to {t.ts.max():%Y-%m-%d}",
        "stats": {"txn_count": int(len(t)), "total_in": round(float(t[t.direction == "in"].amount.sum()), 2),
                  "total_out": round(float(t[t.direction == "out"].amount.sum()), 2),
                  "largest_txn": round(float(t.amount.max()), 2)},
        "risk_score": float(scores.loc[acc, "risk"]),
        "anomaly_percentile": float(scores.loc[acc, "anomaly_pct"]),
        "rule_hits": [{k: h[k] for k in ("rule", "detail", "txn_ids", "count", "total_amount")} for h in hits.get(acc, [])],
        "top_deviations": expl[acc],
        "evidence_txns": ev.assign(ts=ev.ts.dt.strftime("%Y-%m-%d %H:%M"))[
            ["txn_id", "ts", "direction", "channel", "amount", "counterparty_country"]].to_dict("records"),
    }


def template_narrative(c):
    hits = c["rule_hits"]
    if hits:
        rules = ", ".join(h["rule"] for h in hits)
        summary = (f"Account {c['account_id']} ({c['profile']['segment']}, {c['profile']['home_country']}) triggered "
                   f"{len(hits)} typology rule(s): {rules}. Risk score {c['risk_score']:.2f}.")
        flags, action = [h["detail"] for h in hits], "escalate_for_SAR_review"
        cited = [i for h in hits for i in h["txn_ids"] if i in {e["txn_id"] for e in c["evidence_txns"]}][:10]
        typ = hits[0]["rule"]
    else:
        d = c["top_deviations"][0]
        summary = (f"Account {c['account_id']} has no typology rule hit but ranks in the top "
                   f"{(1 - c['anomaly_percentile']) * 100:.0f}% most anomalous accounts vs peers.")
        flags = [f"{x['feature']} = {x['value']} vs peer median {x['peer_median']}" for x in c["top_deviations"]]
        action, typ = "monitor", "anomaly_only"
        cited = [e["txn_id"] for e in c["evidence_txns"][:5]]
    return dict(summary=summary, typology=typ, red_flags=flags, recommended_action=action,
                cited_txn_ids=cited, source="template")


def _nums(s):
    return [float(x.replace(",", "")) for x in re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?", s)]


def validate(n, c):
    """Return a list of grounding problems ([] = every claim traced back to the case packet)."""
    issues = []
    ids = {e["txn_id"] for e in c["evidence_txns"]}
    cited = n.get("cited_txn_ids", [])
    if not cited: issues.append("no transactions cited")
    issues += [f"cited txn {i} is not in the case evidence" for i in cited if i not in ids]
    if n.get("recommended_action") not in ACTIONS: issues.append("recommended_action is not an allowed value")
    text = " ".join([n.get("summary", "")] + list(n.get("red_flags", [])))
    issues += [f"mentions txn {i} not in evidence" for i in set(re.findall(r"T\d{6}", text)) if i not in ids]
    issues += [f"mentions other account {a}" for a in set(re.findall(r"ACC\d{4}", text)) if a != c["account_id"]]
    packet = re.sub(r"\b(T\d{6}|ACC\d{4})\b", "", json.dumps(c, default=str))
    allowed = _nums(packet)
    for num, suffix in re.findall(r"\$\s?(\d+(?:,\d{3})*(?:\.\d+)?)\s?([kKmM])?", text):
        v = float(num.replace(",", "")) * {"": 1, "k": 1e3, "m": 1e6}[suffix.lower()]
        if not any(abs(v - a) <= max(1.0, 0.01 * a) for a in allowed):
            issues.append(f"amount ${num}{suffix} does not appear in the case data")
    return issues


def llm_narrative(c, issues=None):
    import anthropic
    msg = "CASE PACKET:\n" + json.dumps(c, default=str)
    if issues: msg += "\n\nYour previous draft failed validation: " + "; ".join(issues) + ". Fix these."
    r = anthropic.Anthropic().messages.create(model=MODEL, max_tokens=700, system=SYSTEM,
                                              messages=[{"role": "user", "content": msg}])
    text = r.content[0].text
    return json.loads(text[text.index("{"): text.rindex("}") + 1])


def narrative(c):
    """Returns (narrative_dict, issues). Uses Claude if ANTHROPIC_API_KEY is set, with 1 retry, else the template."""
    note = []
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            n = llm_narrative(c); n["source"] = "claude"; issues = validate(n, c)
            if issues:
                n = llm_narrative(c, issues); n["source"] = "claude (retried)"; issues = validate(n, c)
            if not issues: return n, []
            note = ["Claude draft failed grounding twice (" + "; ".join(issues) + ") - showing template instead"]
        except Exception as e:
            note = [f"LLM call failed ({type(e).__name__}) - showing template instead"]
    t = template_narrative(c)
    return t, note + validate(t, c)
