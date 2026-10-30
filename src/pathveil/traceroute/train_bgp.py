#!/usr/bin/env python3
"""Train GMPI with validated inputs, padding masks and an empty-history fallback."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

from .common import (
    assert_source_held_out,
    digest,
    json_write,
    project_imports,
    stable_bgp_split,
    vocab_digest,
    load_contract,
    install_label_guard,
    audit_pool_consumers,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-training-csv", required=True, type=Path)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--data-manifest", required=True, type=Path)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--date", default="20261015")
    parser.add_argument("--source-asn", default="63835")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--k", type=int, default=200)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--walks-type", default="_collectedPath")
    parser.add_argument("--split-only", action="store_true")
    args = parser.parse_args()
    args.generated_training_csv = args.generated_training_csv.resolve()
    args.outdir = args.outdir.resolve()
    if not 0 < args.validation_fraction < 1 or args.epochs < 1 or args.patience < 1:
        parser.error("Invalid split or training limits")
    contract = load_contract(args.data_manifest, args.release, args.source_asn)
    inputs = json.loads(args.input_manifest.read_text())
    if (
        inputs.get("generated_sha256") != digest(args.generated_training_csv)
        or inputs.get("pool_contract") != contract
        or inputs.get("traceroute_labels_read") is not False
    ):
        raise ValueError("Training inputs are not bound to the verified admitted pool")
    args.outdir.mkdir(parents=True, exist_ok=False)
    manifest = stable_bgp_split(
        args.generated_training_csv, args.outdir / "bgp_split", args.validation_fraction, args.seed
    )
    if args.split_only:
        print(json.dumps(manifest, indent=2))
        return
    project_imports(args.data_manifest)
    install_label_guard(args.outdir)
    import numpy as np
    import torch
    import environment
    import GMPI_pathLikelihoodEst as architecture
    import myDataset
    import PathInference_trainer as trainer

    from pathveil.downstream.device_runtime import configure_device

    device, device_info = configure_device(torch, args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed) if torch.cuda.is_available() else None
    if contract["date"] != args.date:
        raise ValueError("Trace routing date and release differ")
    env = environment.Env(args.date, args.date)
    source_audit = assert_source_held_out(env, args.source_asn)
    env.walksType = args.walks_type
    env.cur_vas = args.source_asn
    env.k = args.k
    env.epoch_test_batches = 0
    env.train_loader_params = dict(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False,
        pin_memory=True,
    )
    env.test_loader_params = env.train_loader_params.copy()
    vectors = torch.from_numpy(np.asarray(env.get_node_vectors())).float()
    embedding_dim = env.node2vecOutputDim + env.asrankFeatureDim
    model = architecture.attentionBasedScoreNetwork(
        vectors, embedding_dim * 3, embedding_dim, embedding_dim * 4, args.hidden_size, 20
    ).to(device)
    model.asEmbedding.weight.requires_grad = True
    if model.input_semantics_version != 2:
        raise ValueError("Canonical traceroute training requires model input semantics v2")
    model._pathveil_metadata = {
        "date": args.date,
        "source_asn": args.source_asn,
        "vocab_sha256": vocab_digest(env),
        "k": args.k,
        "max_length": 20,
        "validation": "disjoint BGP-derived source/destination pairs",
        "input_sha256": digest(args.generated_training_csv),
        "seed": args.seed,
        "history_argument_order": "corrected",
        "input_semantics_version": model.input_semantics_version,
        "admitted_path_set_sha256": contract["path_set_sha256"],
        "release_sha256": contract["release_sha256"],
        "data_manifest_sha256": contract["data_manifest_sha256"],
        "train_extra_data_mode": "augmentation_only",
        "pg_local_knowledge_mode": "augmentation_only",
    }
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    datasets = {
        split: myDataset.IterableDatasetShuffle(
            str(args.outdir / "bgp_split" / (split + ".csv")),
            env,
            20,
            args.k,
            60000,
            args.seed,
            False,
        )
        for split in ("train", "validation")
    }
    best, stale = float("-inf"), 0
    checkpoint = args.outdir / "model.pkl"
    history = []
    for epoch in range(args.epochs):
        train_loss, train_reward, train_count, _ = trainer.train(
            model, datasets["train"], optimizer, env, epoch, "train"
        )
        val_loss, val_reward, val_count, _, _, _ = trainer.test(
            model, datasets["validation"], optimizer, env, epoch, "bgp_validation"
        )
        if not train_count or not val_count:
            raise RuntimeError("Empty BGP training or validation dataset")
        score = val_reward / val_count
        row = {
            "epoch": epoch,
            "train_rows": train_count,
            "validation_rows": val_count,
            "train_loss_sum": train_loss,
            "validation_loss_sum": val_loss,
            "train_jaccard_reward": train_reward / train_count,
            "validation_jaccard_reward": score,
        }
        if score > best:
            best, stale = score, 0
            model._pathveil_metadata["selected_epoch"] = epoch
            torch.save(model, checkpoint)
            row["checkpoint_saved"] = True
        else:
            stale += 1
            row["checkpoint_saved"] = False
        history.append(row)
        json_write(args.outdir / "training_history.json", history)
        print(json.dumps(row), flush=True)
        if stale >= args.patience:
            break
    json_write(
        args.outdir / "training_manifest.json",
        {
            "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "source_holdout_audit": source_audit,
            "pool_contract": contract,
            "device": device_info,
            "model_sha256": digest(checkpoint),
            "vocab_sha256": vocab_digest(env),
            "best_validation_jaccard_reward": best,
            "web_test_used_for_selection": False,
            "history_argument_order": "corrected",
            "input_semantics_version": model.input_semantics_version,
            "note": "GMPI parameter dimensions, SmoothL1 objective and BGP-derived labels are preserved. Input semantics v2 validates/deduplicates candidates, masks padding and conditions on query endpoints when history is empty. Full history paths and penultimate hops use explicit corrected arguments. This is validation, not an independent performance claim.",
        },
    )


if __name__ == "__main__":
    main()
