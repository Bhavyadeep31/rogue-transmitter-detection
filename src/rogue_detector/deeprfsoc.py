"""DeepRFSoC import and memory-mapped training pipeline.

The source pickle is a monolithic trusted dataset artifact. It must be
unpickled once, but converted IQ, labels, SNR and split assignments are written
to separate .npy memory maps so later model training stays bounded in RAM.
"""

from __future__ import annotations

import json
import math
import pickle
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def _group_split_ids(count: int, *, seed: int, group_size: int = 512) -> np.ndarray:
    """Stratified 70/15/15 split while keeping contiguous local groups intact."""
    block_count = math.ceil(count / group_size)
    if block_count < 3:
        raise ValueError(f"Need at least three contiguous groups to split a cell; got {block_count}.")
    rng = np.random.default_rng(seed)
    order = rng.permutation(block_count)
    train_blocks = max(1, round(block_count * 0.70))
    validation_blocks = max(1, round(block_count * 0.15))
    if train_blocks + validation_blocks >= block_count:
        validation_blocks = max(1, block_count - train_blocks - 1)
    block_split = np.full(block_count, 2, dtype=np.uint8)
    block_split[order[:train_blocks]] = 0
    block_split[order[train_blocks:train_blocks + validation_blocks]] = 1
    return np.repeat(block_split, group_size)[:count]


def _load_pickle(path: Path) -> dict[Any, Any]:
    # The source is the hash-verified Git LFS object from the official
    # DeepRFSoC repository. Pickle must never be loaded from an untrusted source.
    with path.open("rb") as stream:
        data = pickle.load(stream, encoding="latin1")
    if not isinstance(data, dict):
        raise SystemExit(f"Expected root dict in DeepRFSoC pickle; found {type(data).__name__}.")
    return data


def _validate_and_summarize(data: dict[Any, Any]) -> dict[str, Any]:
    if not data:
        raise SystemExit("DeepRFSoC pickle is empty.")
    classes: set[str] = set()
    snrs: set[int] = set()
    counts: dict[str, dict[str, int]] = {}
    shapes: set[tuple[int, ...]] = set()
    dtypes: set[str] = set()
    total = 0
    for key, values in data.items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise SystemExit(f"Unexpected DeepRFSoC key; expected (modulation, SNR), got {key!r}.")
        modulation, snr_value = str(key[0]), int(key[1])
        if not isinstance(values, np.ndarray) or values.ndim != 3 or values.shape[:2] != (2, 128):
            raise SystemExit(f"Unexpected value for {key!r}: {type(values).__name__}, {getattr(values, 'shape', None)}")
        classes.add(modulation)
        snrs.add(snr_value)
        shapes.add(tuple(int(dim) for dim in values.shape))
        dtypes.add(str(values.dtype))
        counts.setdefault(modulation, {})[str(snr_value)] = int(values.shape[2])
        total += int(values.shape[2])
    shape_prefixes = {(shape[0], shape[1]) for shape in shapes}
    if shape_prefixes != {(2, 128)}:
        raise SystemExit(f"Unsupported DeepRFSoC I/Q/time dimensions: {sorted(shape_prefixes)}")
    if len(dtypes) != 1:
        raise SystemExit(f"Inconsistent DeepRFSoC data dtypes: {sorted(dtypes)}")
    return {
        "root_type": "dict",
        "key_count": len(data),
        "key_format": "(modulation_name: str, snr_db: numeric string)",
        "value_type": "numpy.ndarray",
        "value_shape": [2, 128, "variable_windows_per_key"],
        "distinct_value_shapes": [list(shape) for shape in sorted(shapes)],
        "value_dtype": next(iter(dtypes)),
        "iq_layout": "(I/Q channel, time_sample, window_index); axis 0: I then Q; axis 1: 128 time samples; axis 2: windows",
        "window_count": total,
        "sequence_length": 128,
        "class_names": sorted(classes),
        "class_window_counts": {name: int(sum(counts[name].values())) for name in sorted(classes)},
        "snr_db_values": sorted(snrs),
        "windows_per_class_snr": counts,
        "extra_channel_or_capture_session_metadata": False,
        "available_condition_metadata": ["snr_db in dictionary keys"],
    }


def inspect_deeprfsoc(source_path: str) -> dict[str, Any]:
    """Load and print an exact, bounded summary of the trusted source pickle."""
    source = Path(source_path)
    data = _load_pickle(source)
    summary = _validate_and_summarize(data)
    summary["source_file"] = str(source.resolve())
    summary["source_file_size_bytes"] = source.stat().st_size
    print(json.dumps(summary, indent=2))
    return summary


def import_deeprfsoc(
    source_path: str,
    output_dir: str,
    *,
    seed: int = 7,
    chunk_windows: int = 2048,
    split_group_windows: int = 512,
) -> dict[str, Any]:
    """Convert DeepRFSoC to memory-mapped float32 arrays and grouped splits."""
    source = Path(source_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    data = _load_pickle(source)
    summary = _validate_and_summarize(data)
    class_names = summary["class_names"]
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    keys = sorted(data, key=lambda item: (str(item[0]), int(item[1])))
    total_raw = int(summary["window_count"])

    # First pass counts valid windows without creating a RAM-sized copy.
    valid_count = 0
    nonfinite_count = 0
    for key in keys:
        values = data[key]
        for start in range(0, values.shape[2], chunk_windows):
            chunk = values[:, :, start:start + chunk_windows]
            valid = np.isfinite(chunk).all(axis=(0, 1))
            valid_count += int(valid.sum())
            nonfinite_count += int(valid.size - valid.sum())
    if valid_count == 0:
        raise SystemExit("No finite IQ windows found in the DeepRFSoC pickle.")

    iq_map = np.lib.format.open_memmap(destination / "iq.npy", mode="w+", dtype=np.float32,
                                       shape=(valid_count, 2, 128))
    labels_map = np.lib.format.open_memmap(destination / "labels.npy", mode="w+", dtype=np.uint8,
                                           shape=(valid_count,))
    snr_map = np.lib.format.open_memmap(destination / "snr_db.npy", mode="w+", dtype=np.int16,
                                        shape=(valid_count,))
    split_map = np.lib.format.open_memmap(destination / "split.npy", mode="w+", dtype=np.uint8,
                                          shape=(valid_count,))
    cursor = 0
    cell_counts: dict[str, dict[str, dict[str, int]]] = {}
    for cell_index, key in enumerate(keys):
        modulation, snr_db = str(key[0]), int(key[1])
        values = data.pop(key)
        original_count = int(values.shape[2])
        # Distinct deterministic RNG stream for every class/SNR cell.
        cell_seed = int(np.random.SeedSequence([seed, cell_index]).generate_state(1)[0])
        cell_splits = _group_split_ids(original_count, seed=cell_seed, group_size=split_group_windows)
        before = cursor
        for start in range(0, original_count, chunk_windows):
            stop = min(start + chunk_windows, original_count)
            chunk = values[:, :, start:stop].transpose(2, 0, 1)
            valid = np.isfinite(chunk).all(axis=(1, 2))
            amount = int(valid.sum())
            if amount == 0:
                continue
            end = cursor + amount
            iq_map[cursor:end] = chunk[valid].astype(np.float32, copy=False)
            labels_map[cursor:end] = class_to_id[modulation]
            snr_map[cursor:end] = snr_db
            split_map[cursor:end] = cell_splits[start:stop][valid]
            cursor = end
        cell_counts.setdefault(modulation, {})[str(snr_db)] = {
            "valid_windows": cursor - before,
            "train": int(np.count_nonzero(split_map[before:cursor] == 0)),
            "validation": int(np.count_nonzero(split_map[before:cursor] == 1)),
            "test": int(np.count_nonzero(split_map[before:cursor] == 2)),
        }
        del values

    for mapped in (iq_map, labels_map, snr_map, split_map):
        mapped.flush()
    del iq_map, labels_map, snr_map, split_map

    metadata = {
        **summary,
        "source_file": str(source.resolve()),
        "source_file_size_bytes": source.stat().st_size,
        "processed_windows": int(cursor),
        "nonfinite_windows_dropped": int(nonfinite_count),
        "processed_iq_shape": [int(cursor), 2, 128],
        "processed_iq_dtype": "float32",
        "class_mapping": class_to_id,
        "snr_metadata_path": "snr_db.npy",
        "split_metadata_path": "split.npy; 0=train, 1=validation, 2=test",
        "per_cell_split_counts": cell_counts,
        "preprocessing": {
            "layout_conversion": "(2, 128, windows) -> (windows, 2, 128)",
            "conversion": "float64 to float32; no amplitude normalization applied during import",
            "validity": "discard a window if any I/Q value is nonfinite",
            "storage": "uncompressed NumPy .npy memory maps",
        },
        "split": {
            "seed": seed,
            "group_size_windows": split_group_windows,
            "method": "contiguous window groups kept intact within each (class, SNR) cell; group assignment randomized and stratified by every cell",
            "target_proportions": {"train": 0.70, "validation": 0.15, "test": 0.15},
            "limitation": "No acquisition/session IDs are present; contiguous grouping reduces adjacent-window leakage but cannot guarantee session-disjoint splits.",
        },
        "capture_quality_and_band_occupancy": {
            "available": False,
            "reason": "This corpus contains labeled 128-sample modulation windows and SNR only, with no continuous wideband captures, timestamps, center frequency, sample rate, ADC full-scale/clipping metadata, or N78/occupancy labels.",
        },
    }
    (destination / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    del data
    print("DeepRFSoC source structure:")
    print(json.dumps({key: metadata[key] for key in (
        "root_type", "key_count", "key_format", "value_type", "value_shape", "value_dtype",
        "iq_layout", "window_count", "processed_windows", "sequence_length", "class_names",
        "class_window_counts", "snr_db_values", "extra_channel_or_capture_session_metadata",
    )}, indent=2))
    print(f"Memory-mapped arrays saved under {destination.resolve()}")
    return metadata


def _model_class(torch: Any, nn: Any) -> Any:
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

        def forward(self, values: Any) -> Any:
            return self.activation(values + self.layers(values))

    class IQConvNet(nn.Module):
        def __init__(self, class_count: int) -> None:
            super().__init__()
            self.stem = nn.Sequential(nn.Conv1d(2, 64, 7, padding=3, bias=False),
                                      nn.BatchNorm1d(64), nn.ReLU())
            self.blocks = nn.Sequential(ResidualBlock(64), ResidualBlock(64))
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.35), nn.Linear(64, class_count))

        def forward(self, values: Any) -> Any:
            return self.head(self.pool(self.blocks(self.stem(values))))

    return IQConvNet


def _cell_capped_indices(labels: np.ndarray, snr: np.ndarray, indices: np.ndarray,
                         cap: int, seed: int) -> np.ndarray:
    if cap <= 0:
        return indices
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    cells = np.stack((labels[indices], snr[indices]), axis=1)
    for cell in np.unique(cells, axis=0):
        cell_indices = indices[(cells == cell).all(axis=1)]
        if cell_indices.size > cap:
            cell_indices = rng.choice(cell_indices, cap, replace=False)
        selected.append(cell_indices)
    return np.concatenate(selected) if selected else np.empty(0, dtype=np.int64)


def train_deeprfsoc(
    dataset_dir: str,
    model_path: str,
    metrics_path: str,
    *,
    epochs: int = 20,
    batch_size: int = 512,
    patience: int = 5,
    seed: int = 7,
    max_train_per_cell: int = 0,
    max_eval_per_cell: int = 0,
    resume_from: str | None = None,
    latest_checkpoint_path: str | None = None,
    device_name: str = "auto",
    precision: str = "auto",
) -> dict[str, Any]:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
    except ImportError as error:
        raise SystemExit("DeepRFSoC CNN training requires PyTorch in the active environment.") from error

    directory = Path(dataset_dir)
    metadata = json.loads((directory / "dataset_metadata.json").read_text(encoding="utf-8"))
    iq = np.load(directory / "iq.npy", mmap_mode="r")
    labels = np.load(directory / "labels.npy", mmap_mode="r")
    snr = np.load(directory / "snr_db.npy", mmap_mode="r")
    split = np.load(directory / "split.npy", mmap_mode="r")
    class_names = metadata["class_names"]

    train_idx = np.flatnonzero(split == 0)
    validation_idx = np.flatnonzero(split == 1)
    test_idx = np.flatnonzero(split == 2)
    train_idx = _cell_capped_indices(labels, snr, train_idx, max_train_per_cell, seed)
    validation_idx = _cell_capped_indices(labels, snr, validation_idx, max_eval_per_cell, seed + 1)
    test_idx = _cell_capped_indices(labels, snr, test_idx, max_eval_per_cell, seed + 2)
    if min(len(train_idx), len(validation_idx), len(test_idx)) == 0:
        raise SystemExit("Train/validation/test split is empty.")

    class IQMemmapDataset(Dataset):
        # With automatic batching disabled, the sampler passes a vector of row
        # IDs here. Advanced indexing reads a whole batch from the memmaps at
        # once and avoids millions of Python __getitem__ calls.
        def __getitem__(self, row_ids: np.ndarray) -> tuple[Any, Any, Any]:
            row_ids = np.asarray(row_ids, dtype=np.int64)
            samples = np.asarray(iq[row_ids], dtype=np.float32).copy()
            targets = np.asarray(labels[row_ids], dtype=np.int64)
            conditions = np.asarray(snr[row_ids], dtype=np.int16)
            return torch.from_numpy(samples), torch.from_numpy(targets), torch.from_numpy(conditions)

    class RowBatchSampler:
        def __init__(self, row_ids: np.ndarray, batch: int, *, shuffle: bool, sampler_seed: int) -> None:
            self.row_ids = row_ids
            self.batch = batch
            self.shuffle = shuffle
            self.sampler_seed = sampler_seed
            self.epoch = 0

        def __iter__(self):
            order = self.row_ids.copy()
            if self.shuffle:
                np.random.default_rng(self.sampler_seed + self.epoch).shuffle(order)
                self.epoch += 1
            for start in range(0, order.size, self.batch):
                yield order[start:start + self.batch]

        def __len__(self) -> int:
            return int(math.ceil(self.row_ids.size / self.batch))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device_name not in {"auto", "cpu", "cuda"}:
        raise ValueError("device_name must be auto, cpu, or cuda")
    if precision not in {"auto", "amp", "fp32"}:
        raise ValueError("precision must be auto, amp, or fp32")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is not available")
    device = torch.device("cuda" if device_name == "cuda" or (device_name == "auto" and torch.cuda.is_available()) else "cpu")
    train_sampler = RowBatchSampler(train_idx, batch_size, shuffle=True, sampler_seed=seed)
    train_loader = DataLoader(IQMemmapDataset(), sampler=train_sampler, batch_size=None,
                              num_workers=0, pin_memory=device.type == "cuda")
    validation_loader = DataLoader(IQMemmapDataset(), sampler=RowBatchSampler(validation_idx, batch_size,
                                   shuffle=False, sampler_seed=seed + 1), batch_size=None,
                                   num_workers=0, pin_memory=device.type == "cuda")
    test_loader = DataLoader(IQMemmapDataset(), sampler=RowBatchSampler(test_idx, batch_size,
                             shuffle=False, sampler_seed=seed + 2), batch_size=None,
                             num_workers=0, pin_memory=device.type == "cuda")
    IQConvNet = _model_class(torch, nn)
    model = IQConvNet(len(class_names)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-4)
    amp_enabled = device.type == "cuda" and precision != "fp32"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.03)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    best_val_loss = float("inf")
    best_low_snr_accuracy = -1.0
    best_state = None
    best_epoch = 0
    wait = 0
    start_epoch = 1
    if resume_from:
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        if checkpoint.get("class_names") != class_names:
            raise ValueError("Resume checkpoint class mapping does not match this dataset.")
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint.get("scaler_state_dict", {}))
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_loss = float(checkpoint["best_validation_loss"])
        best_low_snr_accuracy = float(checkpoint["best_low_snr_accuracy"])
        best_state = checkpoint["best_state_dict"]
        best_epoch = int(checkpoint.get("best_epoch", 0))
        wait = int(checkpoint["early_stop_wait"])
        train_sampler.epoch = start_epoch - 1
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"])
        if "numpy_rng_state" in checkpoint:
            np.random.set_state(checkpoint["numpy_rng_state"])
        if "python_rng_state" in checkpoint:
            random.setstate(checkpoint["python_rng_state"])
        if device.type == "cuda" and "cuda_rng_state" in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        print(f"Resuming at epoch {start_epoch} from {resume_from}", flush=True)
    print(f"DeepRFSoC training on {device}; windows train/validation/test="
          f"{len(train_idx)}/{len(validation_idx)}/{len(test_idx)}; batch_size={batch_size}; "
          f"train batches/epoch={len(train_loader)}", flush=True)

    def normalize(batch_x: Any) -> Any:
        rms = torch.sqrt(torch.mean(torch.sum(batch_x * batch_x, dim=1), dim=1, keepdim=True))
        return batch_x / rms.clamp_min(1e-8)[:, None, :]

    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, epochs + 1):
        last_epoch = epoch
        model.train()
        total_loss, correct, seen = 0.0, 0, 0
        epoch_start = time.perf_counter()
        for batch_number, (batch_x, batch_y, _) in enumerate(train_loader, start=1):
            batch_x = batch_x.to(device, non_blocking=amp_enabled)
            batch_y = batch_y.to(device, non_blocking=amp_enabled)
            batch_x = normalize(batch_x)
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
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                logits = model(batch_x)
                loss = loss_fn(logits, batch_y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * batch_y.size(0)
            correct += int((logits.argmax(1) == batch_y).sum().item())
            seen += batch_y.size(0)
            if batch_number % 200 == 0:
                elapsed = time.perf_counter() - epoch_start
                print(f"  epoch {epoch:02d}: batch {batch_number}/{len(train_loader)}; "
                      f"{seen} windows; elapsed {elapsed / 60:.1f} min", flush=True)

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        val_truth: list[np.ndarray] = []
        val_predicted: list[np.ndarray] = []
        val_snr_values: list[np.ndarray] = []
        with torch.no_grad():
            for batch_x, batch_y, batch_snr in validation_loader:
                batch_x = normalize(batch_x.to(device, non_blocking=amp_enabled))
                batch_y = batch_y.to(device, non_blocking=amp_enabled)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                    logits = model(batch_x)
                val_loss_sum += float(loss_fn(logits, batch_y).item()) * batch_y.size(0)
                val_truth.append(batch_y.cpu().numpy())
                val_predicted.append(logits.argmax(1).cpu().numpy())
                val_snr_values.append(batch_snr.numpy())
                val_count += batch_y.size(0)
        val_loss = val_loss_sum / val_count
        val_true_array = np.concatenate(val_truth)
        val_pred_array = np.concatenate(val_predicted)
        val_snr_array = np.concatenate(val_snr_values)
        low_mask = val_snr_array <= 0
        low_snr_accuracy = float(np.mean(val_pred_array[low_mask] == val_true_array[low_mask]))
        validation_accuracy = float(np.mean(val_pred_array == val_true_array))
        print(f"Epoch {epoch:02d}/{epochs} - train loss {total_loss / seen:.4f} - train acc {correct / seen:.4f} - "
              f"validation loss {val_loss:.4f} - validation acc {validation_accuracy:.4f} - "
              f"validation low-SNR acc (<=0 dB) {low_snr_accuracy:.4f}", flush=True)
        scheduler.step(val_loss)
        if low_snr_accuracy > best_low_snr_accuracy or (
            low_snr_accuracy == best_low_snr_accuracy and val_loss < best_val_loss
        ):
            best_val_loss = val_loss
            best_low_snr_accuracy = low_snr_accuracy
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping after {patience} epochs without validation loss improvement.", flush=True)
        if latest_checkpoint_path:
            latest_path = Path(latest_checkpoint_path)
            latest_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "state_dict": model.state_dict(), "best_state_dict": best_state,
                "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(), "class_names": class_names,
                "architecture": "IQResidualCNN-v2", "epoch": epoch,
                "best_validation_loss": best_val_loss, "best_low_snr_accuracy": best_low_snr_accuracy,
                "early_stop_wait": wait, "seed": seed,
                "best_epoch": best_epoch,
                "torch_rng_state": torch.get_rng_state(), "numpy_rng_state": np.random.get_state(),
                "python_rng_state": random.getstate(),
                "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                "device": str(device), "precision": precision,
            }, latest_path)
        if wait >= patience:
            break
    if best_state is None:
        raise RuntimeError("No validation checkpoint was produced.")
    model.load_state_dict(best_state)

    truth: list[np.ndarray] = []
    predicted: list[np.ndarray] = []
    with torch.no_grad():
        for batch_x, batch_y, _ in test_loader:
            batch_x = normalize(batch_x.to(device, non_blocking=amp_enabled))
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                logits = model(batch_x)
            truth.append(batch_y.numpy())
            predicted.append(logits.argmax(1).cpu().numpy())
    y_true = np.concatenate(truth)
    y_pred = np.concatenate(predicted)
    class_ids = list(range(len(class_names)))
    report = classification_report(y_true, y_pred, labels=class_ids, target_names=class_names,
                                   output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=class_ids)
    test_snr = np.asarray(snr[test_idx])
    per_snr: dict[str, dict[str, float | int]] = {}
    for snr_value in sorted(int(value) for value in np.unique(test_snr)):
        mask = test_snr == snr_value
        per_snr[str(snr_value)] = {
            "accuracy": float(np.mean(y_pred[mask] == y_true[mask])),
            "windows": int(mask.sum()),
        }
    metrics = {
        "dataset": "DeepRFSoC",
        "test_accuracy": float(np.mean(y_true == y_pred)),
        "validation_low_snr_accuracy_le_0_db_for_selected_checkpoint": float(best_low_snr_accuracy),
        "test_windows": int(y_true.size),
        "per_class": {name: report[name] for name in class_names},
        "confusion_matrix": {"labels": class_names, "rows_true_columns_predicted": matrix.tolist()},
        "accuracy_by_snr_db": per_snr,
        "split_windows": {"train": int(train_idx.size), "validation": int(validation_idx.size), "test": int(test_idx.size)},
        "limitations": [
            "This is modulation classification, not rogue-versus-legitimate transmitter detection.",
            "The dataset has SNR labels but no channel/session IDs, continuous wideband occupancy labels, or N78 center-frequency metadata.",
            "Contiguous group splitting reduces adjacent-window leakage but cannot prove session-disjoint evaluation.",
        ],
        "training_configuration": {
            "architecture": "IQResidualCNN-v2 compatible residual 1D CNN",
            "epochs_requested": epochs,
            "epochs_completed": last_epoch,
            "best_checkpoint_epoch": best_epoch,
            "batch_size": batch_size,
            "patience": patience,
            "seed": seed,
            "device": str(device),
            "mixed_precision": "CUDA float16 autocast with gradient scaling" if amp_enabled else "disabled",
            "precision_requested": precision,
            "optimizer": "AdamW(lr=0.0008, weight_decay=0.0002)",
            "augmentation": ["per-window shared I/Q RMS normalization", "random common phase", "circular timing shift"],
            "max_train_per_class_snr_cell": max_train_per_cell,
            "max_eval_per_class_snr_cell": max_eval_per_cell,
            "checkpoint_selection": "highest validation accuracy at SNR <= 0 dB, with validation loss as tie-breaker",
            "resumed_from": resume_from,
            "latest_checkpoint_path": latest_checkpoint_path,
        },
        "dataset_metadata": metadata,
    }
    destination = Path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "class_names": class_names,
                "architecture": "IQResidualCNN-v2", "input_format": "float32 I/Q, per-window shared RMS normalized",
                "dataset": "DeepRFSoC", "seed": seed, "best_validation_loss": best_val_loss,
                "best_validation_low_snr_accuracy": best_low_snr_accuracy,
                "best_epoch": best_epoch, "epochs_completed": last_epoch,
                "training_configuration": metrics["training_configuration"],
                "metrics_path": metrics_path}, destination)
    metrics_file = Path(metrics_path)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_file.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    _save_svg_plots(metrics, metrics_file.parent)
    print(f"Overall test accuracy: {metrics['test_accuracy']:.4f}", flush=True)
    print("Per-class precision/recall/F1:", flush=True)
    for name in class_names:
        row = metrics["per_class"][name]
        print(f"  {name}: P={row['precision']:.4f} R={row['recall']:.4f} F1={row['f1-score']:.4f} support={row['support']}", flush=True)
    print("Confusion matrix (rows=true, columns=predicted):", flush=True)
    print(matrix, flush=True)
    print("Accuracy by SNR:", flush=True)
    for value, record in per_snr.items():
        print(f"  {int(value):>3} dB: {record['accuracy']:.4f} ({record['windows']} windows)", flush=True)
    print(f"Saved best validation checkpoint: {destination}", flush=True)
    print(f"Saved metrics: {metrics_file}", flush=True)
    del train_loader, validation_loader, test_loader
    for mapped in (iq, labels, snr, split):
        mapped._mmap.close()
    return metrics


def benchmark_deeprfsoc(
    dataset_dir: str, *, batches: int = 40, seed: int = 7,
    batch_sizes: tuple[int, ...] = (64, 128, 256, 384, 512, 768, 1024, 1536, 2048),
) -> dict[str, Any]:
    """Benchmark real memmap input plus CNN train steps with synchronized CUDA timing."""
    try:
        import torch
        from torch import nn
    except ImportError as error:
        raise SystemExit("DeepRFSoC benchmarking requires PyTorch in the active environment.") from error
    directory = Path(dataset_dir)
    iq = np.load(directory / "iq.npy", mmap_mode="r")
    labels = np.load(directory / "labels.npy", mmap_mode="r")
    split = np.load(directory / "split.npy", mmap_mode="r")
    metadata = json.loads((directory / "dataset_metadata.json").read_text(encoding="utf-8"))
    rows = np.flatnonzero(split == 0)
    rng = np.random.default_rng(seed)
    rng.shuffle(rows)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmark device: {device}; train windows available: {len(rows)}", flush=True)
    results: list[dict[str, Any]] = []
    for precision in (("amp", "fp32") if device.type == "cuda" else ("fp32",)):
        for batch_size in batch_sizes:
            torch.manual_seed(seed)
            model = _model_class(torch, nn)(len(metadata["class_names"])).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4)
            amp = precision == "amp"
            scaler = torch.amp.GradScaler("cuda", enabled=amp)
            loss_fn = nn.CrossEntropyLoss()
            selected = rows[:batch_size * (batches + 2)]
            durations: list[float] = []
            failed: str | None = None
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            try:
                model.train()
                for step in range(batches + 2):
                    step_rows = selected[step * batch_size:(step + 1) * batch_size]
                    begin = time.perf_counter()
                    x = torch.from_numpy(np.asarray(iq[step_rows], dtype=np.float32).copy()).to(device, non_blocking=amp)
                    y = torch.from_numpy(np.asarray(labels[step_rows], dtype=np.int64)).to(device, non_blocking=amp)
                    rms = torch.sqrt(torch.mean(torch.sum(x * x, dim=1), dim=1, keepdim=True))
                    x = x / rms.clamp_min(1e-8)[:, None, :]
                    phase = torch.rand(x.size(0), device=device) * (2 * torch.pi)
                    cosine, sine = phase.cos()[:, None], phase.sin()[:, None]
                    i_part, q_part = x[:, 0, :].clone(), x[:, 1, :].clone()
                    x[:, 0, :] = i_part * cosine - q_part * sine
                    x[:, 1, :] = i_part * sine + q_part * cosine
                    shifts = torch.randint(x.size(-1), (x.size(0),), device=device)
                    sample_index = torch.arange(x.size(-1), device=device)[None, :]
                    gather_index = (sample_index - shifts[:, None]) % x.size(-1)
                    x = x.gather(2, gather_index[:, None, :].expand_as(x))
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                        loss = loss_fn(model(x), y)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    scaler.step(optimizer)
                    scaler.update()
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    elapsed = time.perf_counter() - begin
                    if step >= 2:
                        durations.append(elapsed)
                elapsed_sum = sum(durations)
                result = {
                    "batch_size": batch_size, "precision": precision,
                    "timed_batches": len(durations), "examples_per_second": batch_size * len(durations) / elapsed_sum,
                    "seconds_per_batch": elapsed_sum / len(durations),
                    "estimated_epoch_minutes": (len(rows) / batch_size) * elapsed_sum / len(durations) / 60,
                    "peak_vram_mib": round(torch.cuda.max_memory_allocated(device) / 1048576, 1) if device.type == "cuda" else None,
                    "status": "ok",
                }
            except (RuntimeError, MemoryError) as error:
                failed = str(error).splitlines()[0]
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                result = {"batch_size": batch_size, "precision": precision, "status": "failed", "error": failed}
            results.append(result)
            print(json.dumps(result), flush=True)
            del model, optimizer, scaler
            if device.type == "cuda":
                torch.cuda.empty_cache()
    report = {
        "device": str(device), "cuda_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "seed": seed, "benchmark_batches": batches, "batch_sizes": list(batch_sizes),
        "dataset_train_windows": int(rows.size),
        "process_peak_working_set_mib": _process_peak_working_set_mib(),
        "results": results,
    }
    return report


def _process_peak_working_set_mib() -> float | None:
    """Return this process's peak resident working set without a new dependency."""
    try:
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return None
        return round(counters.PeakWorkingSetSize / (1024 * 1024), 1)
    except (AttributeError, OSError):
        return None


def _save_svg_plots(metrics: dict[str, Any], output_dir: Path) -> None:
    """Write lightweight SVG plots without adding a plotting dependency."""
    output_dir.mkdir(parents=True, exist_ok=True)
    names = metrics["confusion_matrix"]["labels"]
    matrix = metrics["confusion_matrix"]["rows_true_columns_predicted"]
    size, left, top, cell = 900, 210, 80, 70
    maximum = max(max(row) for row in matrix) or 1
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="720" viewBox="0 0 {size} 720">',
             '<style>text{font-family:Segoe UI,Arial,sans-serif;font-size:14px}.title{font-size:22px;font-weight:bold}</style>',
             '<text x="30" y="35" class="title">DeepRFSoC test confusion matrix</text>']
    for row, label in enumerate(names):
        y = top + row * cell
        parts.append(f'<text x="{left - 12}" y="{y + 39}" text-anchor="end">{label}</text>')
        for col, value in enumerate(matrix[row]):
            x = left + col * cell
            intensity = int(235 - 170 * value / maximum)
            parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="rgb({intensity},{intensity},{255})" stroke="white"/>')
            parts.append(f'<text x="{x + cell / 2}" y="{y + 39}" text-anchor="middle">{value}</text>')
    parts.append(f'<text x="{left + cell * len(names) / 2}" y="690" text-anchor="middle">Predicted class</text>')
    parts.append('</svg>')
    (output_dir / "confusion_matrix.svg").write_text("\n".join(parts), encoding="utf-8")

    records = metrics["accuracy_by_snr_db"]
    points = [(int(key), float(value["accuracy"])) for key, value in records.items()]
    if not points:
        return
    width, height, x0, y0, plot_w, plot_h = 900, 500, 90, 70, 760, 340
    min_snr, max_snr = min(x for x, _ in points), max(x for x, _ in points)
    span = max(1, max_snr - min_snr)
    coords = [(x0 + (snr - min_snr) * plot_w / span, y0 + (1.0 - acc) * plot_h) for snr, acc in points]
    polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<style>text{font-family:Segoe UI,Arial,sans-serif;font-size:14px}.title{font-size:22px;font-weight:bold}</style>',
             '<text x="30" y="35" class="title">DeepRFSoC test accuracy by SNR</text>']
    for tick in (0, .25, .5, .75, 1):
        y = y0 + (1 - tick) * plot_h
        parts.append(f'<line x1="{x0}" y1="{y}" x2="{x0 + plot_w}" y2="{y}" stroke="#dddddd"/>')
        parts.append(f'<text x="{x0 - 12}" y="{y + 5}" text-anchor="end">{tick:.0%}</text>')
    parts.extend([f'<polyline points="{polyline}" fill="none" stroke="#1667a8" stroke-width="3"/>',
                  f'<text x="{x0 + plot_w / 2}" y="{height - 20}" text-anchor="middle">SNR (dB)</text>',
                  '<text x="22" y="250" transform="rotate(-90 22 250)" text-anchor="middle">Accuracy</text>'])
    for (snr_value, accuracy), (x, y) in zip(points, coords):
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#b33939"/>')
        parts.append(f'<text x="{x:.1f}" y="{y - 10:.1f}" text-anchor="middle">{accuracy:.2f}</text>')
        parts.append(f'<text x="{x:.1f}" y="{y0 + plot_h + 22}" text-anchor="middle">{snr_value}</text>')
    parts.append('</svg>')
    (output_dir / "accuracy_by_snr.svg").write_text("\n".join(parts), encoding="utf-8")
