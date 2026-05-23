from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def build_model(model_name: str, repo: Path, dropout: float = 0.4):
    sys.path.insert(0, str(repo))
    from models.gnn_models import GCN, GAT, GIN, GraphSAGE, GGNN
    try:
        from models.hgcn_models import HGCN
    except Exception:
        from models.hgnn_models import HGCN
    if model_name == "GCN":
        return GCN(dropout, in_channels=768, hidden_channels=512)
    if model_name == "GAT":
        return GAT(dropout, in_channels=768, hidden_channels=512)
    if model_name == "GIN":
        return GIN(dropout, in_channels=768, hidden_channels=512)
    if model_name == "GraphSAGE":
        return GraphSAGE(dropout, in_channels=768, hidden_channels=512)
    if model_name == "GGNN":
        return GGNN(dropout, in_channels=768, out_channels=512)
    if model_name == "HGCN":
        return HGCN(dropout, in_channels=768, hidden_channels=512)
    raise ValueError(f"Unsupported model: {model_name}")


def compute_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    labeled = [r for r in rows if r.get("true_label") is not None and str(r.get("true_label")) != ""]
    if not labeled:
        return {"mode": "prediction_only", "samples": len(rows)}
    y_true = [int(r["true_label"]) for r in labeled]
    y_pred = [int(r.get("prediction", 0)) for r in labeled]
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    accuracy = (tp + tn) / max(1, len(y_true))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {
        "mode": "evaluated",
        "samples": len(rows),
        "labeled_samples": len(labeled),
        "accuracy": round(accuracy, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def write_outputs(rows: List[Dict[str, Any]], output: Path, csv_path: Path, metrics_path: Path | None) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    fields = sorted({k for r in rows for k in r.keys()}) or ["sample_id", "prediction", "confidence"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    if metrics_path:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(compute_metrics(rows), indent=2, ensure_ascii=False), encoding="utf-8")


def run_real_hgcn(args) -> List[Dict[str, Any]]:
    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))

    import torch
    import torch.nn.functional as F
    from torch_geometric.loader import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[hgcn] device={device}")
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"HCPG dataset not found: {data_path}")
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"HGCN checkpoint not found: {checkpoint}")

    dataset = torch.load(str(data_path), map_location="cpu", weights_only=False)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=0)

    model = build_model(args.model, repo, args.dropout).to(device)
    state = torch.load(str(checkpoint), map_location=device)
    # Accept both raw state_dict and checkpoint dict.
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()

    rows: List[Dict[str, Any]] = []
    idx_base = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            prob = F.softmax(out, dim=1)
            pred = out.argmax(dim=1)
            y = getattr(batch, "y", None)
            batch_size = int(pred.numel())
            for i in range(batch_size):
                true_label = None
                if y is not None:
                    try:
                        true_label = int(y.view(-1)[i].detach().cpu().item())
                    except Exception:
                        true_label = None
                pred_i = int(pred[i].detach().cpu().item())
                confidence = float(prob[i, pred_i].detach().cpu().item())
                vuln_prob = float(prob[i, 1].detach().cpu().item()) if prob.shape[1] > 1 else confidence
                rows.append({
                    "sample_id": idx_base + i,
                    "prediction": pred_i,
                    "vulnerability_probability": round(vuln_prob, 6),
                    "confidence": round(confidence, 6),
                    "true_label": true_label,
                    "risk": "high" if vuln_prob >= 0.80 else "medium" if vuln_prob >= 0.50 else "low",
                    "validation": "model_only",
                    "detector": args.model,
                })
            idx_base += batch_size
    return rows


def run_fallback(args, reason: Exception) -> List[Dict[str, Any]]:
    try:
        from .codebert_common import load_function_dataset, fallback_predictions
    except Exception:
        from codebert_common import load_function_dataset, fallback_predictions
    fallback_data = Path(args.fallback_data) if args.fallback_data else Path(args.data)
    rows = load_function_dataset(fallback_data)
    preds = fallback_predictions(rows, source="fallback_lexical_not_hgcn")
    for r in preds:
        r["fallback_reason"] = str(reason)
    return preds


def main():
    parser = argparse.ArgumentParser(description="SP4CPG HGCN trained-model inference helper")
    parser.add_argument("--repo", required=True, help="SP4CPG repository root")
    parser.add_argument("--model", required=True, choices=["GCN", "GAT", "GIN", "GraphSAGE", "GGNN", "HGCN"])
    parser.add_argument("--checkpoint", default="", help="Path to .pt checkpoint")
    parser.add_argument("--data", required=True, help="Path to hcpg_dataset.pkl")
    parser.add_argument("--output", required=True, help="JSON output path")
    parser.add_argument("--csv", required=True, help="CSV output path")
    parser.add_argument("--metrics", default="", help="Optional metrics JSON path")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--allow-fallback", action="store_true", help="Only for UI demo compatibility; marks detector as fallback_lexical_not_hgcn")
    parser.add_argument("--fallback-data", default="", help="Raw JSON/JSONL dataset for fallback mode")
    args = parser.parse_args()

    try:
        rows = run_real_hgcn(args)
    except Exception as exc:
        if not args.allow_fallback:
            raise
        print(f"[hgcn:fallback] real HGCN prediction is unavailable: {exc}")
        print("[hgcn:fallback] generating marked fallback predictions so the client workflow can finish.")
        rows = run_fallback(args, exc)

    write_outputs(rows, Path(args.output), Path(args.csv), Path(args.metrics) if args.metrics else None)
    print(f"[hgcn] predictions={args.output}")
    print(f"[hgcn] csv={args.csv}")
    if args.metrics:
        print(f"[hgcn] metrics={args.metrics}")


if __name__ == "__main__":
    main()
