"""Inspection helpers for the OAI 5G PLKG capture dataset."""

from __future__ import annotations

import json
import struct
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


QAM_LABELS = {2: "QPSK", 4: "16QAM", 6: "64QAM", 8: "256QAM"}
PROFILE_FEATURE_NAMES = [
    f"{part}_{name}"
    for part in ("iq", "channel_estimate")
    for name in (
        "log10_rms",
        "amplitude_cv",
        "amplitude_p05_over_rms",
        "amplitude_p50_over_rms",
        "amplitude_p95_over_rms",
        "amplitude_p99_over_rms",
        "crest_factor",
        "normalized_mean_real",
        "normalized_mean_imag",
        "pseudo_circularity",
        "phase_coherence",
        "zero_fraction",
        "symbol_power_cv",
    )
]


def _parse_pdsch_file(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    offset = 0
    versions: Counter[int] = Counter()
    qam_order: Counter[int] = Counter()
    symbol_count: Counter[int] = Counter()
    rb_count: Counter[int] = Counter()
    re_shapes: Counter[str] = Counter()
    imsi_ids: set[str] = set()
    empty_imsi = 0
    iq_complex_samples = 0

    while offset < len(data):
        if offset + 5 > len(data) or data[offset : offset + 4] != b"PDSC":
            raise ValueError(f"Invalid PDSCH record magic/header in {path} at byte {offset}.")
        version = data[offset + 4]
        header_size = {3: 53, 4: 61}.get(version)
        if header_size is None or offset + header_size > len(data):
            raise ValueError(f"Unsupported PDSCH format version {version} in {path} at byte {offset}.")

        rb_size = struct.unpack_from("<H", data, offset + 18)[0]
        num_symbols = data[offset + 21]
        qam = data[offset + 26]
        num_rx = data[offset + 28]
        re_per_symbol = struct.unpack_from("<I", data, offset + 29)[0]
        chest_bytes = struct.unpack_from("<I", data, offset + 33)[0]
        iq_bytes = num_rx * num_symbols * re_per_symbol * 4
        record_size = header_size + iq_bytes + chest_bytes
        if (
            num_rx < 1
            or num_symbols < 1
            or re_per_symbol < 1
            or rb_size < 1
            or re_per_symbol != 12 * rb_size
            or chest_bytes % 4
            or offset + record_size > len(data)
        ):
            raise ValueError(f"Invalid PDSCH record dimensions/length in {path} at byte {offset}.")

        imsi = data[offset + 37 : offset + 53].rstrip(b"\0").decode("ascii", "replace")
        if imsi:
            imsi_ids.add(imsi)
        else:
            empty_imsi += 1
        versions[version] += 1
        qam_order[qam] += 1
        symbol_count[num_symbols] += 1
        rb_count[rb_size] += 1
        re_shapes[f"{num_rx}x{num_symbols}x{re_per_symbol}"] += 1
        iq_complex_samples += iq_bytes // 4
        offset += record_size

    return {
        "records": sum(versions.values()),
        "format_versions": dict(sorted(versions.items())),
        "qam_mod_order_counts": dict(sorted(qam_order.items())),
        "qam_class_counts": {
            QAM_LABELS.get(key, f"unknown-Qm-{key}"): count for key, count in sorted(qam_order.items())
        },
        "num_symbols_counts": dict(sorted(symbol_count.items())),
        "allocated_rb_counts": dict(sorted(rb_count.items())),
        "iq_layout_counts_num_rx_symbols_re_per_symbol": dict(sorted(re_shapes.items())),
        "iq_dtype": "interleaved little-endian int16 (I,Q); converted complex samples are complex64",
        "iq_complex_samples": iq_complex_samples,
        "unique_nonempty_imsi_count": len(imsi_ids),
        "empty_imsi_records": empty_imsi,
        "episode_id": path.parent.name,
    }


def _parse_pusch_file(path: Path) -> dict[str, Any]:
    versions: Counter[int] = Counter()
    qam_order: Counter[int] = Counter()
    symbol_count: Counter[int] = Counter()
    rb_count: Counter[int] = Counter()
    re_shapes: Counter[str] = Counter()
    imsi_ids: set[str] = set()
    empty_imsi = 0
    iq_complex_samples = 0

    with path.open("rb") as handle:
        header = handle.read(64)
        if len(header) != 64 or struct.unpack_from("<I", header, 0)[0] != 0x50555343:
            raise ValueError(f"Invalid PUSCH file header in {path}.")
        file_version = struct.unpack_from("<I", header, 4)[0]
        expected_records = struct.unpack_from("<I", header, 12)[0]
        parsed_records = 0
        while True:
            record_header = handle.read(148)
            if not record_header:
                break
            if len(record_header) != 148:
                raise ValueError(f"Truncated PUSCH record header in {path}.")
            record_size = struct.unpack_from("<I", record_header, 0)[0]
            num_symbols = struct.unpack_from("<i", record_header, 36)[0]
            rb_size = struct.unpack_from("<i", record_header, 40)[0]
            re_per_symbol = struct.unpack_from("<i", record_header, 84)[0]
            iq_bytes = struct.unpack_from("<i", record_header, 124)[0]
            chest_bytes = struct.unpack_from("<i", record_header, 144)[0]
            qam = record_header[30]
            imsi = record_header[128:144].rstrip(b"\0").decode("ascii", "replace")
            if (
                record_size != 148 + iq_bytes + chest_bytes
                or num_symbols < 1
                or rb_size < 1
                or re_per_symbol != 12 * rb_size
                or iq_bytes != num_symbols * re_per_symbol * 4
                or chest_bytes < 0
            ):
                raise ValueError(f"Invalid PUSCH record dimensions/length in {path} at record {parsed_records}.")
            if imsi:
                imsi_ids.add(imsi)
            else:
                empty_imsi += 1
            versions[file_version] += 1
            qam_order[qam] += 1
            symbol_count[num_symbols] += 1
            rb_count[rb_size] += 1
            re_shapes[f"{num_symbols}x{re_per_symbol}"] += 1
            iq_complex_samples += iq_bytes // 4
            handle.seek(iq_bytes + chest_bytes, 1)
            parsed_records += 1
        if parsed_records != expected_records:
            raise ValueError(f"PUSCH header says {expected_records} records but parsed {parsed_records} in {path}.")

    return {
        "records": parsed_records,
        "format_versions": dict(sorted(versions.items())),
        "qam_mod_order_counts": dict(sorted(qam_order.items())),
        "qam_class_counts": {
            QAM_LABELS.get(key, f"unknown-Qm-{key}"): count for key, count in sorted(qam_order.items())
        },
        "num_symbols_counts": dict(sorted(symbol_count.items())),
        "allocated_rb_counts": dict(sorted(rb_count.items())),
        "iq_layout_counts_num_symbols_re_per_symbol": dict(sorted(re_shapes.items())),
        "iq_dtype": "interleaved little-endian int16 (I,Q); converted complex samples are complex64",
        "iq_complex_samples": iq_complex_samples,
        "unique_nonempty_imsi_count": len(imsi_ids),
        "empty_imsi_records": empty_imsi,
        "file_format_version": next(iter(versions), None),
    }


def _inspect_pdsch_campaign(campaign_dir: Path) -> dict[str, Any]:
    files = sorted(campaign_dir.rglob("pdsch_dataset.bin"))
    aggregate: Counter[str] = Counter()
    format_versions: Counter[int] = Counter()
    qam_order: Counter[int] = Counter()
    symbol_count: Counter[int] = Counter()
    rb_count: Counter[int] = Counter()
    shapes: Counter[str] = Counter()
    imsi_counts: Counter[int] = Counter()
    empty_imsi = 0
    examples: list[dict[str, Any]] = []
    nonempty_episodes = 0
    empty_episodes: list[str] = []
    for file_path in files:
        parsed = _parse_pdsch_file(file_path)
        if parsed["records"]:
            nonempty_episodes += 1
        else:
            empty_episodes.append(file_path.parent.name)
        aggregate["records"] += parsed["records"]
        aggregate["iq_complex_samples"] += parsed["iq_complex_samples"]
        empty_imsi += parsed["empty_imsi_records"]
        imsi_counts[parsed["unique_nonempty_imsi_count"]] += 1
        format_versions.update({int(k): v for k, v in parsed["format_versions"].items()})
        qam_order.update({int(k): v for k, v in parsed["qam_mod_order_counts"].items()})
        symbol_count.update({int(k): v for k, v in parsed["num_symbols_counts"].items()})
        rb_count.update({int(k): v for k, v in parsed["allocated_rb_counts"].items()})
        shapes.update(parsed["iq_layout_counts_num_rx_symbols_re_per_symbol"])
        if len(examples) < 3:
            examples.append({
                "path": str(file_path),
                "episode_id": parsed["episode_id"],
                "records": parsed["records"],
                "format_versions": parsed["format_versions"],
                "qam_class_counts": parsed["qam_class_counts"],
                "iq_layout_counts": parsed["iq_layout_counts_num_rx_symbols_re_per_symbol"],
            })
    return {
        "episodes": len(files),
        "nonempty_episodes": nonempty_episodes,
        "empty_episode_files": empty_episodes,
        "records": aggregate["records"],
        "format_versions": dict(sorted(format_versions.items())),
        "qam_mod_order_counts": dict(sorted(qam_order.items())),
        "qam_class_counts": {
            QAM_LABELS.get(key, f"unknown-Qm-{key}"): count for key, count in sorted(qam_order.items())
        },
        "num_symbols_counts": dict(sorted(symbol_count.items())),
        "allocated_rb_counts": dict(sorted(rb_count.items())),
        "iq_layout_counts_num_rx_symbols_re_per_symbol": dict(sorted(shapes.items())),
        "iq_dtype": "interleaved little-endian int16 (I,Q); converted complex samples are complex64",
        "iq_complex_samples": aggregate["iq_complex_samples"],
        "empty_imsi_records": empty_imsi,
        "episodes_by_unique_nonempty_imsi_count": dict(sorted(imsi_counts.items())),
        "example_episodes": examples,
    }


def _inspect_paired_h5(path: Path) -> dict[str, Any]:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("PLKG HDF5 inspection requires h5py; install with `python -m pip install -e \".[datasets]\"`.") from exc
    with h5py.File(path, "r") as handle:
        pair_names = sorted(name for name in handle.keys() if name.startswith("pair_"))
        sample: dict[str, Any] = {}
        if pair_names:
            group = handle[pair_names[0]]
            for name, value in group.items():
                if isinstance(value, h5py.Dataset):
                    sample[name] = {"shape": list(value.shape), "dtype": str(value.dtype)}
        return {"path": str(path), "file_bytes": path.stat().st_size, "pairs": len(pair_names), "first_pair_datasets": sample}


def inspect_plkg(input_root: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    """Inspect OAI PLKG binary and paired-HDF5 records without loading IQ payloads all at once."""
    root = Path(input_root)
    if not root.is_dir():
        raise FileNotFoundError(f"PLKG dataset root does not exist: {root}")
    required = [root / "pusch", root / "pdsch", root / "paired"]
    missing = [str(path) for path in required if not path.is_dir()]
    if missing:
        raise FileNotFoundError("PLKG dataset is missing required directories: " + ", ".join(missing))

    campaigns: dict[str, Any] = {}
    pusch_total = 0
    pdsch_total = 0
    pusch_qam: Counter[int] = Counter()
    pdsch_qam: Counter[int] = Counter()
    pusch_imsi_blank = 0
    pdsch_imsi_blank = 0
    pusch_iq_samples = 0
    pdsch_iq_samples = 0

    for path in sorted((root / "pusch").glob("*.bin")):
        result = _parse_pusch_file(path)
        campaigns[path.stem] = {"pusch": result}
        pusch_total += result["records"]
        pusch_imsi_blank += result["empty_imsi_records"]
        pusch_iq_samples += result["iq_complex_samples"]
        pusch_qam.update({int(k): v for k, v in result["qam_mod_order_counts"].items()})

    for path in sorted((root / "pdsch").iterdir()):
        if path.is_dir():
            result = _inspect_pdsch_campaign(path)
            campaigns[path.name] = {**campaigns.get(path.name, {}), "pdsch": result}
            pdsch_total += result["records"]
            pdsch_imsi_blank += result["empty_imsi_records"]
            pdsch_iq_samples += result["iq_complex_samples"]
            pdsch_qam.update({int(k): v for k, v in result["qam_mod_order_counts"].items()})

    paired = [_inspect_paired_h5(path) for path in sorted((root / "paired").glob("*.h5"))]
    files = [path for path in root.rglob("*") if path.is_file() and ".git" not in path.parts]
    qam_union = sorted(set(pusch_qam) | set(pdsch_qam))
    report: dict[str, Any] = {
        "dataset": "OAI 5G PLKG Paired Dataset",
        "source_path": str(root.resolve()),
        "source_file_count_excluding_git_metadata": len(files),
        "source_bytes_excluding_git_metadata": sum(path.stat().st_size for path in files),
        "file_type_counts": {
            ext or "[no extension]": sum((path.suffix.lower() or "[no extension]") == (ext or "[no extension]") for path in files)
            for ext in sorted({path.suffix.lower() for path in files})
        },
        "published_rf_context": {
            "radio": "5G NR SA OAI testbed; n78; PCI 1",
            "campaign_1_2": "106 PRB; strict DMRS-comb filtering; handshake/RRC setup phase",
            "campaign_3": "51 PRB; different physical deployment/site; relaxed PDSCH DMRS filtering",
            "campaign_4": "51 PRB; same site as campaign 3; DMRS filtering relaxed; connected-mode captures included",
            "exact_rf_center_frequency_hz": None,
            "sample_rate_hz": None,
            "snr_db_labels": None,
            "note": "The README and per-record headers identify n78 but do not provide a precise center frequency or ADC sample rate. Do not treat the band label alone as proof that each capture is within 3.3–3.7 GHz.",
        },
        "pusch": {
            "records": pusch_total,
            "iq_complex_samples": pusch_iq_samples,
            "qam_mod_order_counts": dict(sorted(pusch_qam.items())),
            "qam_class_counts": {QAM_LABELS.get(key, f"unknown-Qm-{key}"): count for key, count in sorted(pusch_qam.items())},
            "empty_imsi_records": pusch_imsi_blank,
            "iq_layout": "per-record variable [num_symbols, resource_elements] complex IQ, little-endian int16 I/Q pairs; plus a same-sized channel-estimate block",
            "iq_is_full_wideband_adc_capture": False,
        },
        "pdsch": {
            "records": pdsch_total,
            "iq_complex_samples": pdsch_iq_samples,
            "qam_mod_order_counts": dict(sorted(pdsch_qam.items())),
            "qam_class_counts": {QAM_LABELS.get(key, f"unknown-Qm-{key}"): count for key, count in sorted(pdsch_qam.items())},
            "empty_imsi_records": pdsch_imsi_blank,
            "iq_layout": "per-record variable [num_rx, num_symbols, resource_elements] complex IQ, little-endian int16 I/Q pairs; plus a same-sized channel-estimate block",
            "iq_is_full_wideband_adc_capture": False,
        },
        "paired_hdf5": paired,
        "campaigns": campaigns,
        "model_training_assessment": {
            "unique_qam_labels": [QAM_LABELS.get(key, f"unknown-Qm-{key}") for key in qam_union],
            "has_noise_or_non_nr_negative_examples": False,
            "has_rogue_or_authorized_transmitter_classes": False,
            "supports_supervised_nr_vs_background_training": False,
            "supports_supervised_modulation_classification": len(qam_union) > 1,
            "recommendation": "Use this as n78 NR positive/profile data. It cannot train or validate a supervised NR-vs-background or legitimate-vs-rogue classifier by itself. All observed PUSCH/PDSCH records have Qm=2 (QPSK); the dataset also lacks noise, non-NR, and rogue labels. Add matched negative/background captures before training a detector.",
        },
    }
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _complex_summary(values: np.ndarray, num_symbols: int) -> list[float]:
    values = np.asarray(values, dtype=np.complex64).reshape(-1)
    magnitude = np.abs(values).astype(np.float32, copy=False)
    rms = float(np.sqrt(np.mean(magnitude * magnitude, dtype=np.float64))) if magnitude.size else 0.0
    scale = max(rms, 1e-12)
    normalized = values / scale
    normalized_amplitude = magnitude / scale
    quantiles = np.quantile(normalized_amplitude, [0.05, 0.50, 0.95, 0.99]) if magnitude.size else np.zeros(4)
    phase_unit = values / np.maximum(magnitude, 1e-12)
    symbol_power_cv = 0.0
    if num_symbols > 0 and values.size % num_symbols == 0:
        per_symbol = values.reshape(num_symbols, -1)
        row_power = np.mean(np.abs(per_symbol) ** 2, axis=1)
        row_mean = float(np.mean(row_power))
        if row_mean > 0:
            symbol_power_cv = float(np.std(row_power) / row_mean)
    result = [
        float(np.log10(max(rms, 1e-12))),
        float(np.std(normalized_amplitude)),
        *[float(value) for value in quantiles],
        float(np.max(normalized_amplitude)) if magnitude.size else 0.0,
        float(np.mean(normalized.real)) if values.size else 0.0,
        float(np.mean(normalized.imag)) if values.size else 0.0,
        float(abs(np.mean(normalized * normalized))) if values.size else 0.0,
        float(abs(np.mean(phase_unit))) if values.size else 0.0,
        float(np.mean(magnitude == 0)) if magnitude.size else 1.0,
        symbol_power_cv,
    ]
    return result


def _pdsch_capture_features(path: Path):
    """Yield compact per-capture features and non-sensitive scheduling metadata."""
    data = path.read_bytes()
    offset = 0
    while offset < len(data):
        version = data[offset + 4]
        header_size = {3: 53, 4: 61}[version]
        num_symbols = data[offset + 21]
        start_symbol = data[offset + 20]
        num_rx = data[offset + 28]
        re_per_symbol = struct.unpack_from("<I", data, offset + 29)[0]
        chest_bytes = struct.unpack_from("<I", data, offset + 33)[0]
        iq_bytes = num_rx * num_symbols * re_per_symbol * 4
        iq_offset = offset + header_size
        chest_offset = iq_offset + iq_bytes
        iq_iq = np.frombuffer(data, dtype="<i2", count=iq_bytes // 2, offset=iq_offset).reshape(-1, 2)
        chest_iq = np.frombuffer(data, dtype="<i2", count=chest_bytes // 2, offset=chest_offset).reshape(-1, 2)
        iq_complex = iq_iq[:, 0].astype(np.float32) + 1j * iq_iq[:, 1].astype(np.float32)
        chest_complex = chest_iq[:, 0].astype(np.float32) + 1j * chest_iq[:, 1].astype(np.float32)
        features = _complex_summary(iq_complex, num_symbols) + _complex_summary(chest_complex, num_symbols)
        frame = struct.unpack_from("<H", data, offset + 9)[0]
        slot = data[offset + 11]
        rb_start = struct.unpack_from("<H", data, offset + 16)[0]
        rb_size = struct.unpack_from("<H", data, offset + 18)[0]
        bwp_start = struct.unpack_from("<H", data, offset + 14)[0]
        yield features, {
            "frame": frame,
            "slot": slot,
            "qam_mod_order": data[offset + 26],
            "num_symbols": num_symbols,
            "start_symbol": start_symbol,
            "num_rx": num_rx,
            "rb_start": rb_start,
            "rb_size": rb_size,
            "bwp_start": bwp_start,
        }
        offset += header_size + iq_bytes + chest_bytes


def import_plkg_profile(input_root: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Stream PDSCH records into compact feature arrays for a one-class profile baseline."""
    root = Path(input_root)
    report = inspect_plkg(root)
    total = int(report["pdsch"]["records"])
    if total == 0:
        raise ValueError("No PDSCH captures were found in this PLKG dataset.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    feature_path = output / "pdsch_profile_features.npy"
    feature_map = np.lib.format.open_memmap(
        feature_path, mode="w+", dtype=np.float32, shape=(total, len(PROFILE_FEATURE_NAMES))
    )
    metadata: dict[str, list[Any]] = {
        "campaign": [], "episode_id": [], "frame": [], "slot": [], "qam_mod_order": [],
        "num_symbols": [], "start_symbol": [], "num_rx": [], "rb_start": [], "rb_size": [], "bwp_start": [],
    }
    row = 0
    counts_by_campaign: Counter[str] = Counter()
    for campaign_dir in sorted(path for path in (root / "pdsch").iterdir() if path.is_dir()):
        for path in sorted(campaign_dir.rglob("pdsch_dataset.bin")):
            for features, fields in _pdsch_capture_features(path):
                if row >= total:
                    raise ValueError("PLKG PDSCH record count changed between inspection and import.")
                feature_map[row] = features
                metadata["campaign"].append(campaign_dir.name)
                metadata["episode_id"].append(f"{campaign_dir.name}/{path.parent.name}")
                for key, value in fields.items():
                    metadata[key].append(value)
                row += 1
                counts_by_campaign[campaign_dir.name] += 1
    feature_map.flush()
    del feature_map
    if row != total:
        raise ValueError(f"Expected {total} PDSCH records but imported {row}.")
    metadata_arrays = {
        key: np.asarray(values, dtype="<U64" if key in {"campaign", "episode_id"} else None)
        for key, values in metadata.items()
    }
    np.savez_compressed(output / "pdsch_profile_metadata.npz", **metadata_arrays)
    preprocessing = {
        "input": str(root.resolve()),
        "direction": "PDSCH downlink only",
        "record_count": row,
        "feature_names": PROFILE_FEATURE_NAMES,
        "feature_layout": "26 float32 summary features: 13 from each record's allocated frequency-domain IQ and OAI channel-estimate block",
        "grouping": "campaign/episode, used to prevent capture-session leakage",
        "per_campaign_record_counts": dict(counts_by_campaign),
        "nonempty_episode_count": int(np.unique(metadata_arrays["episode_id"]).size),
        "source_rf_context": report["published_rf_context"],
        "caveat": "This is a one-class profile feature set, not full-band raw ADC IQ and not an NR-vs-background or rogue-labeled dataset.",
    }
    (output / "metadata.json").write_text(json.dumps(preprocessing, indent=2), encoding="utf-8")
    return preprocessing


def train_plkg_profile(
    dataset_dir: str | Path,
    model_path: str | Path,
    metrics_path: str | Path,
    *,
    seed: int = 7,
    n_estimators: int = 250,
    validation_percentile: float = 95.0,
    smoke_max_records_per_split: int = 0,
) -> dict[str, Any]:
    """Fit a one-class PDSCH profile; outputs false-alert rates, never rogue accuracy."""
    directory = Path(dataset_dir)
    features = np.load(directory / "pdsch_profile_features.npy", mmap_mode="r")
    metadata = np.load(directory / "pdsch_profile_metadata.npz", allow_pickle=False)
    campaigns = metadata["campaign"]
    episodes = metadata["episode_id"]
    known_campaigns = sorted(np.unique(campaigns).tolist())
    if not known_campaigns:
        raise ValueError("No campaign labels were found in the imported PLKG metadata.")
    rng = np.random.default_rng(seed)
    validation_groups: set[str] = set()
    test_groups: set[str] = set()
    for campaign in known_campaigns:
        group_names = np.unique(episodes[(campaigns == campaign)])
        if group_names.size < 3:
            raise ValueError(f"Need at least three capture episodes in {campaign} for a group split.")
        shuffled = rng.permutation(group_names)
        n_validation = max(1, int(round(group_names.size * 0.15)))
        n_test = max(1, int(round(group_names.size * 0.15)))
        validation_groups.update(shuffled[:n_validation].tolist())
        test_groups.update(shuffled[n_validation : n_validation + n_test].tolist())
    validation_mask = np.isin(episodes, list(validation_groups))
    test_mask = np.isin(episodes, list(test_groups))
    train_mask = ~(validation_mask | test_mask)
    if np.any(train_mask & validation_mask) or np.any(train_mask & test_mask) or np.any(validation_mask & test_mask):
        raise RuntimeError("PLKG profile split has capture/campaign leakage.")

    def select(mask: np.ndarray) -> np.ndarray:
        indexes = np.flatnonzero(mask)
        if smoke_max_records_per_split and indexes.size > smoke_max_records_per_split:
            indexes = np.sort(rng.choice(indexes, size=smoke_max_records_per_split, replace=False))
        return indexes

    train_idx = select(train_mask)
    validation_idx = select(validation_mask)
    test_idx = select(test_mask)
    fit_budget_per_campaign = 4096
    fit_rng = np.random.default_rng(seed + 1)
    fit_chunks: list[np.ndarray] = []
    for campaign in known_campaigns:
        candidates = train_idx[campaigns[train_idx] == campaign]
        if candidates.size == 0:
            continue
        fit_chunks.append(np.sort(fit_rng.choice(
            candidates,
            size=min(fit_budget_per_campaign, len(candidates)) if len(candidates) >= fit_budget_per_campaign else fit_budget_per_campaign,
            replace=len(candidates) < fit_budget_per_campaign,
        )))
    fit_idx = np.concatenate(fit_chunks) if fit_chunks else np.empty(0, dtype=np.int64)
    fit_campaign_counts = {campaign: int(np.sum(campaigns[fit_idx] == campaign)) for campaign in known_campaigns}
    estimator = make_pipeline(
        StandardScaler(),
        IsolationForest(
            n_estimators=n_estimators,
            max_samples=min(4096, len(fit_idx)),
            contamination="auto",
            random_state=seed,
            n_jobs=-1,
        ),
    )
    estimator.fit(np.asarray(features[fit_idx], dtype=np.float32))
    val_scores = -estimator.decision_function(np.asarray(features[validation_idx], dtype=np.float32))
    threshold = float(np.quantile(val_scores, validation_percentile / 100.0))
    test_scores = -estimator.decision_function(np.asarray(features[test_idx], dtype=np.float32))
    test_flags = test_scores > threshold
    per_campaign: dict[str, Any] = {}
    for campaign in sorted(np.unique(campaigns[test_idx]).tolist()):
        local = campaigns[test_idx] == campaign
        per_campaign[campaign] = {
            "records": int(local.sum()),
            "known_profile_false_alert_rate": float(np.mean(test_flags[local])),
            "anomaly_score_median": float(np.median(test_scores[local])),
            "anomaly_score_p95": float(np.quantile(test_scores[local], 0.95)),
        }
    result: dict[str, Any] = {
        "task": "one-class known-NR PDSCH profile novelty baseline",
        "model_type": "StandardScaler + IsolationForest",
        "seed": seed,
        "n_estimators": n_estimators,
        "max_samples": min(4096, len(fit_idx)),
        "model_fit_record_count": int(len(fit_idx)),
        "model_fit_records_by_campaign": fit_campaign_counts,
        "validation_threshold_percentile": validation_percentile,
        "anomaly_score_threshold": threshold,
        "split_policy": "campaign-stratified 70/15/15 split by episode; no episode is shared across train/validation/test",
        "split_episode_counts": {
            "train": int(np.unique(episodes[train_idx]).size),
            "validation": int(np.unique(episodes[validation_idx]).size),
            "test": int(np.unique(episodes[test_idx]).size),
        },
        "split_record_counts": {"train": int(len(train_idx)), "validation": int(len(validation_idx)), "test": int(len(test_idx))},
        "validation_known_profile_false_alert_rate": float(np.mean(val_scores > threshold)),
        "heldout_known_profile_false_alert_rate": float(np.mean(test_flags)),
        "heldout_by_campaign": per_campaign,
        "classification_accuracy": None,
        "rogue_detection_recall": None,
        "interpretation": "False-alert rates measure how often held-out captures from the same OAI testbed profile cross a novelty threshold. No rogue/non-NR samples exist here, so these are not rogue detection metrics.",
        "features": PROFILE_FEATURE_NAMES,
        "training_limitations": [
            "All PDSCH records have QPSK modulation order and belong to the same OAI testbed cell profile.",
            "The inputs are frequency-domain allocated resource elements and channel estimates, not wideband ADC recordings.",
            "The dataset does not specify exact RF center frequency/sample rate per campaign or include noise/non-NR/rogue labels.",
        ],
    }
    model_file = Path(model_path)
    metrics_file = Path(metrics_path)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, model_file)
    metrics_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result
