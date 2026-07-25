#!/usr/bin/env python3
"""Verify a CRA5 checkpoint against the Artemka-style formal checklist.

Usage:
  PYTHONPATH=src python scripts/verify_checkpoint.py [--name cra5_era5_28ch_best]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from era5_minimum.cra5.model import build_cra5_model


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def assert_file(p: Path, what: str, fail: list[str]) -> None:
    if p.exists():
        return
    fail.append(f"missing {what}: {p}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="cra5_era5_28ch_best")
    ap.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    ap.add_argument("--artifacts-dir", type=Path, default=ROOT / "artifacts")
    args = ap.parse_args()

    ckpt = args.checkpoint_dir / f"{args.name}.pth"
    prov = args.artifacts_dir / f"{args.name}.json"
    stats = args.artifacts_dir / f"{args.name}_train_stats.json"
    splits = args.artifacts_dir / f"{args.name}_splits.json"
    ev = args.artifacts_dir / f"{args.name}_eval.json"

    fails: list[str] = []
    warn: list[str] = []
    results = []

    for p, w in [
        (ckpt, "checkpoint .pth"),
        (prov, "provenance .json"),
        (stats, "train stats .json"),
        (splits, "splits .json"),
        (ev, "eval .json"),
    ]:
        assert_file(p, w, fails)

    if ckpt.exists():
        bundle = torch.load(ckpt, map_location="cpu", weights_only=False)
        cfg = bundle["config"]
        results.append(("weights load correctly (torch.load)", True))
        results.append((f"architecture class = {cfg.get('class_name','?')}", cfg.get("class_name") == "Cra5Vaeformer28"))

        required_cfg = ["in_channels", "out_channels", "hidden_dim", "latent_dim", "patch_size", "channels"]
        for k in required_cfg:
            results.append((f"config has {k}", k in cfg))

        results.append(("28 in_channels", cfg.get("in_channels") == 28))
        results.append(("28 out_channels", cfg.get("out_channels") == 28))
        results.append(("28 channels list", len(cfg.get("channels", [])) == 28))

        norm = bundle.get("normalization", {})
        results.append(("norm mean = 28", len(norm.get("mean", [])) == 28))
        results.append(("norm std  = 28", len(norm.get("std", [])) == 28))
        results.append(("norm channels = 28", len(norm.get("channels", [])) == 28))

        meta = bundle.get("training_metadata", {})
        for k in ("git_commit", "seed", "best_epoch", "global_step_total", "best_val_loss_latitude_weighted"):
            results.append((f"training_meta has {k}", k in meta))

        rb = meta.get("resource_budget", {})
        if rb:
            results.append(
                ("trainable params ≤ 20M",
                 int(rb.get("trainable_params_actual", 0)) <= 20_000_000),
            )
            results.append(
                ("opt steps ≤ 50 000",
                 int(rb.get("global_steps_actual", 0)) <= 50_000),
            )
            results.append(("extra pretraining forbidden flag", rb.get("extra_pretraining_forbidden") is True))

        # Strict load_state_dict
        try:
            model = build_cra5_model(
                in_channels=cfg["in_channels"],
                out_channels=cfg["out_channels"],
                hidden_dim=cfg["hidden_dim"],
                latent_dim=cfg.get("latent_dim"),
                num_encoder_blocks=cfg["num_encoder_blocks"],
                num_decoder_blocks=cfg["num_decoder_blocks"],
                num_heads=cfg["num_heads"],
                patch_size=tuple(cfg["patch_size"]),
            )
            missing, unexpected = model.load_state_dict(bundle["model_state_dict"], strict=True)
            results.append(
                (f"strict load_state_dict (missing={len(missing)}, unexpected={len(unexpected)})",
                 len(missing) == 0 and len(unexpected) == 0),
            )
        except Exception as e:  # noqa: BLE001
            results.append((f"strict load_state_dict raised: {e}", False))

        # SHA + size
        actual_sha = sha256(ckpt)
        actual_size = ckpt.stat().st_size
        if prov.exists():
            pv = json.loads(prov.read_text())
            results.append(("provenance sha256 == actual", pv.get("sha256") == actual_sha))
            results.append(("provenance size_bytes == actual", int(pv.get("size_bytes", -1)) == actual_size))
        else:
            warn.append("skip sha/size check: provenance json missing")

    if stats.exists():
        s = json.loads(stats.read_text())
        results.append(("stats fit_on = train_split_only", s.get("fit_on") == "train_split_only"))
        results.append(("stats method = per_channel_zscore", s.get("method") == "per_channel_zscore"))
        if "sst_ocean_policy" in s:
            results.append(("stats declares SST/ocean mask policy", True))
        else:
            warn.append("sst_ocean_policy missing in train_stats (demo placeholder expected)")

    if splits.exists():
        sp = json.loads(splits.read_text())
        seeds = [sp["split_seeds"][k]["seed"] for k in ("train", "val", "test")]
        results.append(("3 distinct split seeds", len(set(seeds)) == 3))
        results.append(("split temporal_leakage_policy declared", "temporal_leakage_policy" in sp))

    if ev.exists():
        e = json.loads(ev.read_text())
        results.append(("eval split == test", e.get("split_evaluated") == "test"))
        results.append(("eval has physical_rmse_overall", "physical_rmse_overall" in e))
        results.append(("eval has 28 per-channel rmses",
                         len(e.get("physical_rmse_per_channel", {})) == 28))

    print("=" * 72)
    print(f"Checklist verification:  {args.name}")
    print("=" * 72)
    passed = 0
    total = len(results)
    for txt, ok in results:
        tag = "✅" if ok else "❌"
        if ok:
            passed += 1
        print(f"  {tag}  {txt}")
    if fails:
        print()
        print("FATAL MISSING FILES:")
        for f in fails:
            print(f"  ❌  {f}")
    if warn:
        print()
        print("WARNINGS:")
        for w in warn:
            print(f"  ⚠️  {w}")
    print()
    print(f"SUMMARY:  {passed}/{total + len(fails)} checks passed ({int(100*passed/max(1,total+len(fails)))}%)")
    print()
    return 0 if (passed == total + len(fails)) else 2


if __name__ == "__main__":
    sys.exit(main())
