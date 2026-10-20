# AML Alert Copilot

Transaction-monitoring pipeline with a GenAI case-summary layer and a human-in-the-loop review UI.
Built on **synthetic data** - it demonstrates the pipeline design, not real-world detection accuracy.

## Pipeline
```
transactions -> typology rules (evidence txn ids) ─┐
             -> IsolationForest anomaly score ─────┴-> hybrid risk score -> ranked alert queue
alert -> case packet (facts only) -> Claude drafts narrative -> validator checks every txn id / $ figure
      -> analyst reviews, picks disposition -> audit log (data/decisions.csv)
```

## Run
```
pip install -r requirements.txt
python generate_data.py      # synthetic data + ground-truth labels
python eval.py               # metrics vs baselines + grounding tests
streamlit run app.py         # analyst UI
export ANTHROPIC_API_KEY=... # optional: Claude narratives (offline template otherwise)
```

## Design choices (and why)
- **Rules + anomaly model**: rules are explainable and map to known typologies (structuring, rapid pass-through, high-risk geography); the unsupervised model is meant to surface behaviour no rule encodes.
- **Evidence-first**: every rule hit carries the txn ids behind it, so each alert is auditable.
- **LLM never sees raw data or decides**: it gets a compact case packet, drafts text only, and a deterministic validator rejects any txn id or dollar figure not in the packet (one retry, then falls back to a template).
- **Human-in-the-loop**: nothing is auto-filed; analyst disposition + rationale are logged.

## Known limitations
- Data and rules were written by the same person, so rules-only scoring perfectly on `eval.py` is circular - it proves the plumbing, not generalisation.
- Anomaly layer is untested on typologies the rules don't cover (see next steps).
- Account-level only: no customer graph / network analysis, no KYC features, no time-evolving thresholds.
- Illustrative thresholds and high-risk country list; real deployments use institution policy and official lists.

## Next steps
- Inject a 4th typology with no rule (e.g. dormant account suddenly active) and measure what the anomaly layer recovers.
- Add counterparty-graph features (shared beneficiaries, fan-in/fan-out).
- Analyst-feedback loop: use saved dispositions to tune thresholds.
