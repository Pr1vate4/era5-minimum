#!/usr/bin/env python3
"""Profile CRA5 parameter count — find configs under 20M trainable params."""
from __future__ import annotations
import sys
from pathlib import Path
if str(Path(__file__).resolve().parents[1] / "src") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from era5_minimum.cra5.model import build_cra5_model

configs = [
    ("h1024 l256 8bl", 1024, 256, 8, 8, 16),
    ("h768  l256 6bl", 768, 256, 6, 6, 12),
    ("h512  l192 6bl", 512, 192, 6, 6, 8),
    ("h512  l128 6bl", 512, 128, 6, 6, 8),
    ("h384  l96  6bl", 384, 96, 6, 6, 6),
    ("h384  l96  4bl", 384, 96, 4, 4, 6),
    ("h320  l80  4bl", 320, 80, 4, 4, 5),
    ("h256  l96  6bl", 256, 96, 6, 6, 4),
    ("h256  l64  6bl", 256, 64, 6, 6, 4),
    ("h256  l64  4bl", 256, 64, 4, 4, 4),
    ("h256  l48  4bl", 256, 48, 4, 4, 4),
    ("h192  l48  6bl", 192, 48, 6, 6, 3),
    ("h192  l48  4bl", 192, 48, 4, 4, 3),
    ("h192  l32  4bl", 192, 32, 4, 4, 3),
]

print(f"{'config':<20} {'hidden':>6} {'latent':>6} {'params(M)':>10} {'<20M':>5}")
print("-" * 60)
for name, h, l, eb, db, nh in configs:
    m = build_cra5_model(28, 28, h, l, eb, db, nh, (4,4))
    p = sum(x.numel() for x in m.parameters())
    ok = p <= 20_000_000
    print(f"{name:<20} {h:>6} {l:>6} {p/1e6:>10.2f} {'✅' if ok else '❌':>5}")
