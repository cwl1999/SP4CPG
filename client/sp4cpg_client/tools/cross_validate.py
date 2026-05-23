from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def load_json(path: Path, default: Any):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def normalize_static(static_data: Any) -> Dict[str, Dict[str, Any]]:
    """Return static evidence indexed by sample_id/function/file when possible."""
    index: Dict[str, Dict[str, Any]] = {}
    if isinstance(static_data, dict):
        if isinstance(static_data.get("alerts"), list):
            items = static_data["alerts"]
        elif isinstance(static_data.get("warnings"), list):
            items = static_data["warnings"]
        else:
            items = [static_data]
    elif isinstance(static_data, list):
        items = static_data
    else:
        items = []
    for item in items:
        if not isinstance(item, dict):
            continue
        keys = [item.get("sample_id"), item.get("function"), item.get("function_name"), item.get("file"), item.get("path")]
        for k in keys:
            if k is not None and str(k) != "":
                index[str(k)] = item
    return index


def main():
    parser = argparse.ArgumentParser(description="Cross validate SP4CPG model predictions with static-analysis evidence")
    parser.add_argument("--model-result", required=True)
    parser.add_argument("--static-result", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()

    model_rows = load_json(Path(args.model_result), [])
    static_data = load_json(Path(args.static_result), [])
    if not isinstance(model_rows, list):
        raise ValueError("model-result should be a JSON array")
    static_index = normalize_static(static_data)

    final_rows: List[Dict[str, Any]] = []
    for row in model_rows:
        if not isinstance(row, dict):
            continue
        keys = [row.get("sample_id"), row.get("function"), row.get("function_name"), row.get("file"), row.get("path")]
        evidence = None
        for k in keys:
            if k is not None and str(k) in static_index:
                evidence = static_index[str(k)]
                break
        vuln_prob = float(row.get("vulnerability_probability", row.get("confidence", 0.0)) or 0.0)
        prediction = int(row.get("prediction", 0) or 0)
        if prediction == 1 and evidence:
            validation = "confirmed"
            risk = "high"
            reason = "model_prediction_and_static_evidence"
        elif prediction == 1 and vuln_prob >= 0.80:
            validation = "review_required"
            risk = "medium"
            reason = "high_model_score_without_static_evidence"
        elif prediction == 1:
            validation = "low_confidence"
            risk = "low"
            reason = "model_only_without_static_evidence"
        else:
            validation = "safe_or_unconfirmed"
            risk = "low"
            reason = "model_safe_or_low_score"
        final = dict(row)
        final.update({
            "validation": validation,
            "risk": risk,
            "cross_validation_reason": reason,
            "static_evidence": bool(evidence),
            "static_summary": json.dumps(evidence, ensure_ascii=False)[:500] if evidence else "",
        })
        final_rows.append(final)

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv); csv_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(final_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    fields = sorted({k for r in final_rows for k in r.keys()}) or ["sample_id", "prediction", "validation", "risk"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(final_rows)
    print(f"[cross-validate] model={len(model_rows)} static_evidence={len(static_index)} final={len(final_rows)}")
    print(f"[cross-validate] output={out}")


if __name__ == "__main__":
    main()
