"""1D CNN training/evaluation for RadioML modulation recognition."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def _parse_snr_list(values: str) -> list[int]:
    try:
        return sorted({int(value.strip()) for value in values.split(",") if value.strip()})
    except ValueError as error:
        raise SystemExit("SNR values must be comma-separated integers, e.g. -20,-10,0,10,18") from error


def train_cnn(
    dataset_path: str,
    model_path: str,
    *,
    epochs: int = 40,
    batch_size: int = 256,
    test_size: float = 0.2,
    validation_size: float = 0.15,
    max_examples: int = 0,
    split_mode: str = "per_snr",
    heldout_snrs: str = "-20,-10,0,10,18",
    patience: int = 8,
    seed: int = 7,
) -> None:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as error:
        raise SystemExit(
            "The CNN needs PyTorch. Install it with `python -m pip install -e .[deep-learning]` "
            "using a Python version supported by the current PyTorch release."
        ) from error

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with np.load(dataset_path, allow_pickle=False) as dataset:
        iq = dataset["iq"]
        labels = dataset["labels"].astype(str)
        if "snr_db" not in dataset.files:
            raise SystemExit("SNR-aware training needs an NPZ containing `snr_db`; re-import RadioML data.")
        snr_db = dataset["snr_db"].astype(np.int16)
    if max_examples and max_examples < len(labels):
        rng = np.random.default_rng(seed)
        selected = rng.choice(len(labels), max_examples, replace=False)
        iq, labels, snr_db = iq[selected], labels[selected], snr_db[selected]

    class_names = sorted(np.unique(labels).tolist())
    label_to_id = {name: index for index, name in enumerate(class_names)}
    y = np.asarray([label_to_id[label] for label in labels], dtype=np.int64)
    heldout = _parse_snr_list(heldout_snrs)
    indices = np.arange(len(y))
    if split_mode == "per_snr":
        # Keep each modulation/SNR cell represented in all splits. This answers:
        # how well does the model work at each SNR when it has seen that SNR in training?
        strata = np.asarray([f"{label}:{snr}" for label, snr in zip(y, snr_db)])
        train_pool_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=seed, stratify=strata
        )
        validation_strata = strata[train_pool_idx]
    elif split_mode == "snr_holdout":
        unseen_mask = np.isin(snr_db, heldout)
        if not unseen_mask.any():
            raise SystemExit(f"None of the requested held-out SNR values are in the dataset: {heldout}")
        if np.all(unseen_mask):
            raise SystemExit("All dataset SNR values are held out; leave some SNR values for training.")
        train_pool_idx = np.flatnonzero(~unseen_mask)
        test_idx = np.flatnonzero(unseen_mask)
        validation_strata = y[train_pool_idx]
    else:
        raise SystemExit("split_mode must be 'per_snr' or 'snr_holdout'")
    train_idx, validation_idx = train_test_split(
        train_pool_idx,
        test_size=validation_size,
        random_state=seed,
        stratify=validation_strata,
    )

    # Normalize each window by one shared I/Q RMS value. This removes arbitrary
    # recording gain while preserving relative I/Q geometry and the SNR content.
    x = np.stack((iq.real, iq.imag), axis=1).astype(np.float32)
    rms = np.sqrt(np.mean(np.sum(x * x, axis=1), axis=1, keepdims=True))
    x /= np.maximum(rms[:, None, :], 1e-8)

    train_data = TensorDataset(torch.from_numpy(x[train_idx]), torch.from_numpy(y[train_idx]))
    validation_data = TensorDataset(torch.from_numpy(x[validation_idx]), torch.from_numpy(y[validation_idx]))
    test_data = TensorDataset(torch.from_numpy(x[test_idx]), torch.from_numpy(y[test_idx]))
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=generator)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    class ResidualBlock(nn.Module):
        def __init__(self, channels: int, kernel_size: int = 5) -> None:
            super().__init__()
            padding = kernel_size // 2
            self.layers = nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False),
                nn.BatchNorm1d(channels), nn.ReLU(),
                nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False),
                nn.BatchNorm1d(channels),
            )
            self.activation = nn.ReLU()

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            return self.activation(values + self.layers(values))

    class IQConvNet(nn.Module):
        def __init__(self, class_count: int) -> None:
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv1d(2, 64, kernel_size=7, padding=3, bias=False),
                nn.BatchNorm1d(64), nn.ReLU(),
            )
            self.blocks = nn.Sequential(ResidualBlock(64), ResidualBlock(64))
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.35), nn.Linear(64, class_count))

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            return self.head(self.pool(self.blocks(self.stem(values))))

    model = IQConvNet(len(class_names)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-4)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.03)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, seen = 0.0, 0, 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            # Random common phase and circular timing shift model receiver offsets.
            phase = torch.rand(batch_x.size(0), device=device) * (2 * torch.pi)
            cos_phase, sin_phase = phase.cos()[:, None], phase.sin()[:, None]
            i_part, q_part = batch_x[:, 0, :].clone(), batch_x[:, 1, :].clone()
            batch_x[:, 0, :] = i_part * cos_phase - q_part * sin_phase
            batch_x[:, 1, :] = i_part * sin_phase + q_part * cos_phase
            shifts = torch.randint(batch_x.size(-1), (batch_x.size(0),), device=device)
            sample_index = torch.arange(batch_x.size(-1), device=device)[None, :]
            gather_index = (sample_index - shifts[:, None]) % batch_x.size(-1)
            batch_x = batch_x.gather(2, gather_index[:, None, :].expand_as(batch_x))
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item() * batch_y.size(0)
            correct += int((logits.argmax(1) == batch_y).sum().item())
            seen += batch_y.size(0)

        model.eval()
        val_losses: list[float] = []
        val_true: list[np.ndarray] = []
        val_pred: list[np.ndarray] = []
        with torch.no_grad():
            for batch_x, batch_y in validation_loader:
                logits = model(batch_x.to(device))
                val_losses.append(float(loss_fn(logits, batch_y.to(device)).item()))
                val_true.append(batch_y.numpy())
                val_pred.append(logits.argmax(1).cpu().numpy())
        val_loss = float(np.mean(val_losses))
        val_accuracy = float(np.mean(np.concatenate(val_true) == np.concatenate(val_pred)))
        scheduler.step(val_loss)
        print(
            f"Epoch {epoch:02d}/{epochs} - train loss {total_loss / seen:.4f} - "
            f"train acc {correct / seen:.3f} - validation loss {val_loss:.4f} - "
            f"validation acc {val_accuracy:.3f}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping: validation loss did not improve for {patience} epochs.")
                break

    if best_state is None:
        raise RuntimeError("CNN training did not produce a best validation checkpoint")
    model.load_state_dict(best_state)

    def predict(loader: DataLoader) -> np.ndarray:
        model.eval()
        batches: list[np.ndarray] = []
        with torch.no_grad():
            for batch_x, _ in loader:
                batches.append(model(batch_x.to(device)).argmax(1).cpu().numpy())
        return np.concatenate(batches)

    y_val_pred = predict(validation_loader)
    y_test_pred = predict(test_loader)
    all_labels = np.arange(len(class_names))
    if split_mode == "per_snr":
        print(f"CNN evaluation ({Path(dataset_path).stem}): train/validation/test each contain samples from every SNR.")
    else:
        print(f"CNN evaluation ({Path(dataset_path).stem}): final test uses SNR levels excluded from training.")
        print(f"Held-out SNR levels (excluded from training): {heldout}")
    print("Seen-SNR validation report:")
    print(classification_report(y[validation_idx], y_val_pred, labels=all_labels,
                                target_names=class_names, zero_division=0))
    print("SNR test report:")
    print(classification_report(y[test_idx], y_test_pred, labels=all_labels,
                                target_names=class_names, zero_division=0))
    print("SNR test confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y[test_idx], y_test_pred, labels=all_labels))
    print("Test accuracy by SNR:")
    for value in sorted(np.unique(snr_db[test_idx])):
        snr_indices = test_idx[snr_db[test_idx] == value]
        if snr_indices.size:
            pred_at_snr = predict(DataLoader(
                TensorDataset(torch.from_numpy(x[snr_indices]), torch.from_numpy(y[snr_indices])),
                batch_size=batch_size, shuffle=False,
            ))
            print(f"  {value:>3} dB: {np.mean(pred_at_snr == y[snr_indices]):.3f} ({snr_indices.size} windows)")

    destination = Path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": best_state,
        "class_names": class_names,
        "input_format": "float32 I/Q channels, per-window shared RMS normalized, shape (batch, 2, samples)",
        "training_data_kind": Path(dataset_path).stem,
        "architecture": "IQResidualCNN-v2",
        "split_mode": split_mode,
        "heldout_snrs_db": heldout if split_mode == "snr_holdout" else [],
        "best_validation_loss": best_val_loss,
        "seed": seed,
    }, destination)
    print(f"Saved best validation checkpoint to {destination}")
