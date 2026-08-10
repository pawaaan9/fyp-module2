"""
Train one synthetic-label control GAT.

    python train_synthetic_gat.py --type corner   --sign pos
    python train_synthetic_gat.py --type freekick --sign neg
    python train_synthetic_gat.py --type corner   --sign pos --limit 8000

Same architecture and same featurisation as Module 1's real GAT (both come
from pitchpulse/graph.py), trained on the synthetic geometric labels instead
of match outcomes. Writes:

    models/module2_synthetic_gat_{type}_{sign}.pt

The checkpoint carries the metadata graph.py:load_model verifies -- feature
order, normalisation constants, architecture kwargs -- so these models load
through the ordinary GATScorer path with no special casing.

    NOT A FOOTBALL MODEL. It predicts pitchpulse/synthetic.py's geometry rule.

READING THE LOG
---------------
pred_sd is the standard deviation of the model's predictions on the held-out
set. It is the column that matters. A regression model can reach respectable
loss by ignoring its input and emitting the training mean for every scenario,
and such a model is worthless as a directional control -- every edit returns a
delta of zero, so it can neither confirm nor deny the direction. pred_sd near
zero is that failure. Below 0.02 the run aborts rather than saving a
checkpoint that would look fine in a results table.

Frames are split by scenario_id, never by row: all variants of one real frame
land on the same side of the split, so the test set measures generalisation to
unseen geometry rather than memorisation of a frame the model already saw.
"""

import argparse
import json
import os
import random
import time

SIGNS = {"pos": 1.0, "neg": -1.0}
GRAPH_FEAT_DIM = {"corner": 6, "freekick": 4}
COLLAPSE_THRESHOLD = 0.02


def load_rows(path, limit=None):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def build_dataset(rows, device):
    """Featurise every row once. Returns a list of PyG Data objects with y set."""
    import torch

    from pitchpulse.graph import build_graph

    data_list, skipped = [], 0
    t0 = time.time()
    for i, row in enumerate(rows):
        scenario = {
            "scenario_id": row["scenario_id"],
            "set_piece_type": row["set_piece_type"],
            "kick_location": row.get("kick_location"),
            "players": row["players"],
        }
        try:
            data = build_graph(scenario)
        except Exception:  # noqa: BLE001 -- rows the shared guard rejects
            skipped += 1
            continue
        data.y = torch.tensor([float(row["label"])], dtype=torch.float)
        data.scenario_id = row["scenario_id"]
        data_list.append(data)

        if (i + 1) % 5000 == 0:
            print(f"    featurised {i + 1}/{len(rows)} ({time.time() - t0:.0f}s)")

    if skipped:
        print(f"    skipped {skipped} rows rejected by the graph guard")
    return data_list


def split_by_scenario(data_list, test_frac=0.2, seed=0):
    ids = sorted({d.scenario_id for d in data_list})
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_test = max(int(len(ids) * test_frac), 1)
    test_ids = set(ids[:n_test])
    train = [d for d in data_list if d.scenario_id not in test_ids]
    test = [d for d in data_list if d.scenario_id in test_ids]
    return train, test


def evaluate(model, loader, device, criterion):
    import torch

    model.eval()
    preds, targets, total_loss, n = [], [], 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logit, _, _ = model(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.graph_feat
            )
            logit = logit.view(-1)
            y = batch.y.view(-1)
            total_loss += float(criterion(logit, y)) * y.numel()
            n += y.numel()
            preds.append(torch.sigmoid(logit).cpu())
            targets.append(y.cpu())

    preds = torch.cat(preds)
    targets = torch.cat(targets)

    from scipy.stats import spearmanr

    rho = spearmanr(preds.numpy(), targets.numpy()).statistic
    return {
        "loss": total_loss / max(n, 1),
        "spearman": float(rho),
        "mae": float((preds - targets).abs().mean()),
        "pred_sd": float(preds.std()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--type", choices=sorted(GRAPH_FEAT_DIM), required=True)
    ap.add_argument("--sign", choices=sorted(SIGNS), required=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap on dataset ROWS -- use if training runs out of memory")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden-dim", type=int, default=32)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--outdir", default="models")
    args = ap.parse_args()

    import torch
    from torch_geometric.loader import DataLoader

    from pitchpulse.graph import (
        FEATURE_VERSION,
        GRAPH_FEATURE_NAMES,
        HULL_NORM,
        MARK_CLIP,
        NEAR_GOAL_DIST,
        NODE_FEATURE_NAMES,
        PITCH_LENGTH,
        PITCH_WIDTH,
        _enriched_gat_class,
    )

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)

    data_path = f"output/synthetic_{args.type}_{args.sign}.jsonl"
    meta_path = f"output/synthetic_{args.type}_{args.sign}.meta.json"
    for p in (data_path, meta_path):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} -- run build_synthetic_dataset.py first")

    with open(meta_path) as f:
        meta = json.load(f)

    print(f"=== {args.type} / {args.sign} ===")
    print(f"  data  {data_path}")
    rows = load_rows(data_path, args.limit)
    print(f"  {len(rows)} rows" + (f" (limited from {meta['n_rows']})" if args.limit else ""))

    data_list = build_dataset(rows, device)
    if not data_list:
        raise SystemExit("no usable graphs")

    train_data, test_data = split_by_scenario(data_list, seed=args.seed)
    print(f"  {len(train_data)} train / {len(test_data)} test graphs "
          f"(split by scenario_id, no frame spans both)")

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)

    init_kwargs = {
        "node_dim": len(NODE_FEATURE_NAMES),
        "edge_dim": 1,
        "hidden_dim": args.hidden_dim,
        "heads": args.heads,
        "graph_feat_dim": GRAPH_FEAT_DIM[args.type],
    }
    model = _enriched_gat_class()(**init_kwargs).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    print(f"\n  {'epoch':>5} {'train':>8} {'test':>8} {'spearman':>9} "
          f"{'mae':>7} {'pred_sd':>8}   {'time':>6}")

    best = None
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total, n = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optimiser.zero_grad()
            logit, _, _ = model(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.graph_feat
            )
            loss = criterion(logit.view(-1), batch.y.view(-1))
            loss.backward()
            optimiser.step()
            total += float(loss.detach()) * batch.y.numel()
            n += batch.y.numel()

        stats = evaluate(model, test_loader, device, criterion)
        print(f"  {epoch:5d} {total / max(n, 1):8.4f} {stats['loss']:8.4f} "
              f"{stats['spearman']:9.4f} {stats['mae']:7.4f} {stats['pred_sd']:8.4f}   "
              f"{time.time() - t0:5.1f}s")
        best = stats

    # --- the collapse gate --------------------------------------------------
    if best["pred_sd"] < COLLAPSE_THRESHOLD:
        raise SystemExit(
            f"\nABORT: final test pred_sd {best['pred_sd']:.4f} < {COLLAPSE_THRESHOLD}.\n"
            f"The model has collapsed to predicting the mean, so every counterfactual "
            f"edit would return a delta of zero and the checkpoint is useless as a "
            f"directional control. NOT saving it."
        )

    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(
        args.outdir, f"module2_synthetic_gat_{args.type}_{args.sign}.pt"
    )
    torch.save(
        {
            "model_class": "EnrichedGAT",
            "model_state_dict": model.state_dict(),
            "init_kwargs": init_kwargs,
            "set_piece_type": args.type,
            "node_feat_order": NODE_FEATURE_NAMES,
            "graph_feat_order": GRAPH_FEATURE_NAMES[args.type],
            "feature_version": FEATURE_VERSION,
            "norm_constants": {
                "PITCH_X": PITCH_LENGTH,
                "PITCH_Y": PITCH_WIDTH,
                "HULL_NORM": HULL_NORM,
                "MARK_CLIP": MARK_CLIP,
                "NEAR_GOAL_DIST": NEAR_GOAL_DIST,
            },
            # provenance -- so a stray checkpoint can never be mistaken for Module 1's
            "labels_are_synthetic": True,
            "sign": args.sign,
            "sign_value": SIGNS[args.sign],
            "rule_version": meta["rule_version"],
            "calibration": meta["calibration"],
            "trained_on": data_path,
            "n_rows_used": len(rows),
            "row_limit": args.limit,
            "test_metrics": best,
            "warning": (
                "Synthetic geometric labels, NOT football outcomes. This is a "
                "directional control and must never be reported as a football model."
            ),
        },
        out_path,
    )

    print(f"\n  FINAL TEST  loss {best['loss']:.4f}  spearman {best['spearman']:.4f}  "
          f"mae {best['mae']:.4f}  pred_sd {best['pred_sd']:.4f}")
    print(f"  saved {out_path}")


if __name__ == "__main__":
    main()
