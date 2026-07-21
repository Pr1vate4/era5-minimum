from __future__ import annotations

import random
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class TrainResult:
    train_loss: float
    runtime_seconds: float


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def train_autoencoder(
    model: nn.Module,
    dataset: Dataset[torch.Tensor],
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    num_workers: int = 0,
) -> TrainResult:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    model.to(device)
    model.train()
    started = time.perf_counter()
    last_loss = float("nan")

    for _ in range(epochs):
        running = 0.0
        count = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(batch)
            loss = criterion(reconstruction, batch)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * batch.shape[0]
            count += batch.shape[0]
        last_loss = running / max(count, 1)

    return TrainResult(train_loss=last_loss, runtime_seconds=time.perf_counter() - started)
