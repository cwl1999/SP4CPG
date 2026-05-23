
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

try:
    from .codebert_common import load_function_dataset, fallback_predictions, compute_basic_metrics, write_rows
except Exception:
    from codebert_common import load_function_dataset, fallback_predictions, compute_basic_metrics, write_rows


def train_real_codebert(rows: List[Dict[str, Any]], args) -> List[Dict[str, Any]]:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    labeled = [r for r in rows if r.get("target") is not None]
    if len(labeled) < 4:
        raise ValueError("CodeBERT training requires at least 4 labeled samples. Use prediction or fallback mode for tiny data.")
    random.Random(7).shuffle(labeled)
    split = max(1, int(len(labeled) * 0.8))
    train_rows, test_rows = labeled[:split], labeled[split:]
    if not test_rows:
        test_rows = train_rows[-1:]
        train_rows = train_rows[:-1]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    class FuncDataset(Dataset):
        def __init__(self, items): self.items = items
        def __len__(self): return len(self.items)
        def __getitem__(self, idx):
            row = self.items[idx]
            enc = tokenizer(row["func"], truncation=True, padding="max_length", max_length=args.max_length, return_tensors="pt")
            enc = {k: v.squeeze(0) for k, v in enc.items()}
            enc["labels"] = torch.tensor(int(row["target"]), dtype=torch.long)
            return enc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2).to(device)
    loader = DataLoader(FuncDataset(train_rows), batch_size=args.batch, shuffle=True)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            optim.step(); optim.zero_grad()
            total += float(out.loss.detach().cpu().item())
        print(f"[codebert:train] epoch={epoch+1}/{args.epochs} loss={total/max(1,len(loader)):.6f}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[codebert:train] checkpoint={out_dir}")

    # Reuse prediction helper from saved checkpoint for all rows, so output shape is consistent.
    try:
        from .codebert_predict import run_real_codebert
    except Exception:
        from codebert_predict import run_real_codebert
    return run_real_codebert(rows, args.model_name, str(out_dir), args.max_length)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/fine-tune CodeBERT for function-level vulnerability detection")
    parser.add_argument("--data", required=True)
    parser.add_argument("--model-name", default="microsoft/codebert-base")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--pred-json", required=True)
    parser.add_argument("--pred-csv", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--allow-fallback", action="store_true")
    args = parser.parse_args()

    rows = load_function_dataset(Path(args.data))
    print(f"[codebert:train] samples={len(rows)} model={args.model_name}")
    try:
        preds = train_real_codebert(rows, args)
        metrics = compute_basic_metrics(preds)
        metrics["detector"] = "codebert"
    except Exception as exc:
        if not args.allow_fallback:
            raise
        print(f"[codebert:train:fallback] real CodeBERT training is unavailable: {exc}")
        print("[codebert:train:fallback] writing a fallback checkpoint and predictions so the workflow can finish.")
        out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "fallback_model.json").write_text(json.dumps({"type": "fallback_lexical_not_codebert", "reason": str(exc)}, indent=2), encoding="utf-8")
        preds = fallback_predictions(rows, source="fallback_lexical_not_codebert")
        metrics = compute_basic_metrics(preds)
        metrics["detector"] = "fallback_lexical_not_codebert"
        metrics["fallback_reason"] = str(exc)

    write_rows(preds, Path(args.pred_json), Path(args.pred_csv))
    Path(args.metrics).write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[codebert:train] metrics={args.metrics}")
    print(f"[codebert:train] predictions={args.pred_json}")


if __name__ == "__main__":
    main()
