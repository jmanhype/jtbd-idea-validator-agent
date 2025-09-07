"""
Tiny GEPA optimizer for the Judge module.

Usage:
  python -m jtbd_agent_dspy.tools.optimize_judge --train data/judge_train.jsonl --out artifacts/judge_compiled.dspy

Input format (JSONL):
  {"summary": "...", "scorecard": {"criteria":[{"name":"Underserved Opportunity","score":7.0,"rationale":"..."}, ...], "total": 6.7}}

This compiles an improved JudgeScore program using DSPy GEPA and saves it.
You can then load it at runtime by setting JTBD_JUDGE_COMPILED=artifacts/judge_compiled.dspy
"""
import os, json, argparse, dspy, pathlib, pickle
from typing import List, Dict
from dspy.teleprompt import GEPA
from plugins.llm_dspy import JudgeScoreSig, CRITERIA, configure_lm

def load_examples(path: str):
    xs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            rec = json.loads(line)
            summ = rec.get("summary","")
            sc   = rec.get("scorecard",{})
            # normalize criteria to known set
            crits = []
            for it in sc.get("criteria", []):
                n = it.get("name")
                if n in CRITERIA:
                    crits.append({"name": n, "score": float(it.get("score",5.0)), "rationale": it.get("rationale","")})
            # fill missing
            present = {c["name"] for c in crits}
            for n in CRITERIA:
                if n not in present:
                    crits.append({"name": n, "score": 5.0, "rationale": "defaulted"})
            total = round(sum(c["score"] for c in crits)/len(crits), 2)
            gold = {"criteria": crits, "total": total}
            xs.append(dspy.Example(summary=summ, scorecard_json=json.dumps(gold)).with_inputs("summary"))
    return xs

def non_decreasing_metric(example, pred, trace=None, pred_name=None, pred_trace=None):
    """Returns 1 if predicted total >= gold total, else 0."""
    try:
        import json as _json
        p = _json.loads(pred.scorecard_json)
        g = _json.loads(example.scorecard_json)
        return 1.0 if p.get("total",0) >= g.get("total",0) else 0.0
    except Exception:
        return 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="Path to JSONL of training examples")
    ap.add_argument("--out", required=True, help="Output path for compiled program (.dspy)")
    ap.add_argument("--budget", choices=["light", "medium", "heavy"], default="light", help="GEPA budget")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    configure_lm()
    prog = dspy.Predict(JudgeScoreSig)  # base program
    train = load_examples(args.train)
    tele = GEPA(metric=non_decreasing_metric, auto=args.budget)
    compiled = tele.compile(prog, trainset=train)
    with open(args.out, "wb") as f:
        pickle.dump(compiled, f)
    print(f"Wrote {args.out} with {len(train)} examples.")

if __name__ == "__main__":
    main()
