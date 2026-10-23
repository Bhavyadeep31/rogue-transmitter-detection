"""Command-line entry point for the prototype."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from rogue_detector.features import extract_features
from rogue_detector.cnn import train_cnn
from rogue_detector.deeprfsoc import benchmark_deeprfsoc, inspect_deeprfsoc, import_deeprfsoc, train_deeprfsoc
from rogue_detector.plkg import import_plkg_profile, inspect_plkg, train_plkg_profile
from rogue_detector.synthetic import CLASS_NAMES, DEFAULT_SAMPLE_RATE_HZ, generate_dataset


def _import_radioml(args: argparse.Namespace) -> None:
    """Convert the RadioML2016.10A dictionary into the project's NPZ layout."""
    source = Path(args.input)
    # RadioML is a pickle file. Only load a copy obtained from the trusted Kaggle
    # dataset page; pickle files can execute code when opened.
    with source.open("rb") as handle:
        try:
            data = pickle.load(handle, encoding="latin1")
        except TypeError:
            handle.seek(0)
            data = pickle.load(handle)

    requested = set(args.modulations.split(",")) if args.modulations else None
    keys = sorted(data, key=lambda item: (str(item[0]), int(item[1])))
    chosen = [key for key in keys if requested is None or str(key[0]) in requested]
    if not chosen:
        raise SystemExit("No modulation keys selected; check --modulations spelling.")

    iq_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    snr_parts: list[np.ndarray] = []
    class_names: list[str] = []
    rng = np.random.default_rng(args.seed)
    for key in chosen:
        modulation, snr = str(key[0]), int(key[1])
        examples = np.asarray(data[key])
        if examples.ndim != 3 or examples.shape[1] != 2:
            raise SystemExit(f"Unexpected RadioML array shape for {key}: {examples.shape}")
        if args.max_per_class_snr and examples.shape[0] > args.max_per_class_snr:
            selection = rng.choice(examples.shape[0], args.max_per_class_snr, replace=False)
            examples = examples[selection]
        iq = (examples[:, 0, :] + 1j * examples[:, 1, :]).astype(np.complex64)
        iq_parts.append(iq)
        label_parts.append(np.full(iq.shape[0], modulation, dtype=f"U{max(8, len(modulation))}"))
        snr_parts.append(np.full(iq.shape[0], snr, dtype=np.int16))
        if modulation not in class_names:
            class_names.append(modulation)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    iq = np.concatenate(iq_parts)
    labels = np.concatenate(label_parts)
    snr_db = np.concatenate(snr_parts)
    permutation = rng.permutation(labels.size)
    np.savez_compressed(output, iq=iq[permutation], labels=labels[permutation], snr_db=snr_db[permutation])
    metadata = {
        "dataset_kind": "RadioML2016.10A",
        "source": "DeepSig dataset via Kaggle mirror",
        "source_url": "https://www.kaggle.com/datasets/nolasthitnotomorrow/radioml2016-deepsigcom",
        "class_names": class_names,
        "examples": int(labels.size),
        "samples_per_window": int(iq.shape[1]),
        "class_counts": {name: int(np.count_nonzero(labels == name)) for name in class_names},
        "snr_db_values": sorted(int(value) for value in np.unique(snr_db)),
        "license": "Verify the Kaggle dataset page and underlying DeepSig terms before redistribution.",
        "limitations": "Modulation recognition benchmark labels; not 5G N78-specific and not rogue/legitimate transmitter labels.",
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Imported {labels.size} IQ windows with {len(class_names)} modulation classes to {output}")
    print(f"Classes: {', '.join(class_names)}")


def _import_radiomod16(args: argparse.Namespace) -> None:
    """Convert the public RadioMod-R16 HDF5 arrays into the project's NPZ layout."""
    try:
        import h5py
    except ImportError as error:
        raise SystemExit("Install the dataset reader with `python -m pip install -e .[datasets]`.") from error

    source = Path(args.input)
    requested = {name.strip().upper() for name in args.modulations.split(",") if name.strip()}
    with h5py.File(source, "r") as dataset:
        iq_values = dataset["X"][:]
        one_hot = dataset["Y"][:]
        snr_db = dataset["Z"][:].astype(np.int16)
    if iq_values.ndim != 3 or iq_values.shape[1:] != (512, 2) or one_hot.ndim != 2:
        raise SystemExit(f"Unexpected RadioMod-R16 array shapes: X={iq_values.shape}, Y={one_hot.shape}")

    all_names = ["BPSK", "QPSK", "8PSK", "OQPSK", "DQPSK", "PI4QPSK", "16QAM", "32QAM",
                 "64QAM", "16APSK", "2FSK", "4FSK", "MSK", "CPFSK", "GMSK", "GFSK"]
    if one_hot.shape[1] != len(all_names):
        raise SystemExit(f"Unexpected RadioMod-R16 class count: {one_hot.shape[1]}")
    ids = one_hot.argmax(axis=1)
    labels_all = np.asarray(all_names)[ids]
    selected_mask = np.isin(labels_all, sorted(requested))
    if not selected_mask.any():
        raise SystemExit(f"No requested modulation classes found: {sorted(requested)}")

    iq_selected = iq_values[selected_mask]
    iq = (iq_selected[:, :, 0] + 1j * iq_selected[:, :, 1]).astype(np.complex64)
    labels = labels_all[selected_mask]
    snr_db = snr_db[selected_mask]
    rng = np.random.default_rng(args.seed)
    permutation = rng.permutation(labels.size)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, iq=iq[permutation], labels=labels[permutation], snr_db=snr_db[permutation])
    metadata = {
        "dataset_kind": "RadioMod-R16",
        "source_url": "https://zenodo.org/records/19045423",
        "license": "CC BY 4.0",
        "class_names": sorted(np.unique(labels).tolist()),
        "examples": int(labels.size),
        "samples_per_window": int(iq.shape[1]),
        "snr_db_values": sorted(int(value) for value in np.unique(snr_db)),
        "limitations": "Simulated modulation-recognition data with channel impairments; not N78-specific and has no rogue/legitimate transmitter labels.",
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Imported {labels.size} RadioMod-R16 windows, classes={metadata['class_names']}, to {output}")


def _train_radioml(args: argparse.Namespace) -> None:
    """Evaluate the existing feature baseline against a RadioML NPZ subset."""
    dataset_path = Path(args.dataset)
    with np.load(dataset_path, allow_pickle=False) as dataset:
        iq = dataset["iq"]
        labels = dataset["labels"].astype(str)
        snr_db = dataset["snr_db"] if "snr_db" in dataset.files else None
    if args.max_examples and args.max_examples < len(labels):
        rng = np.random.default_rng(args.seed)
        selected = rng.choice(len(labels), args.max_examples, replace=False)
        iq, labels = iq[selected], labels[selected]
        if snr_db is not None:
            snr_db = snr_db[selected]
    features = np.vstack([extract_features(window) for window in iq])
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=args.test_size, random_state=args.seed, stratify=labels
    )
    classifier = RandomForestClassifier(
        n_estimators=args.trees, class_weight="balanced", random_state=args.seed, n_jobs=-1
    )
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)
    class_names = sorted(np.unique(labels).tolist())
    print("RadioML modulation-recognition holdout (not rogue transmitter detection)")
    print(classification_report(y_test, predictions, labels=class_names, zero_division=0))
    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y_test, predictions, labels=class_names))
    if snr_db is not None:
        test_indices = train_test_split(
            np.arange(len(labels)), test_size=args.test_size, random_state=args.seed, stratify=labels
        )[1]
        print("Accuracy by SNR (same random stratified holdout):")
        for value in sorted(np.unique(snr_db[test_indices])):
            mask = snr_db[test_indices] == value
            print(f"  {int(value):>3} dB: {np.mean(predictions[mask] == y_test[mask]):.3f}")
    model_path = Path(args.model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump({"classifier": classifier, "class_names": class_names,
                 "feature_version": "iq-spectrum-v1", "training_data_kind": "RadioML2016.10A"}, model_path)
    print(f"Saved model to {model_path}")


def _generate(args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    iq, labels = generate_dataset(
        examples_per_class=args.examples,
        n_samples=args.samples_per_window,
        sample_rate_hz=args.sample_rate,
        seed=args.seed,
    )
    np.savez_compressed(output, iq=iq, labels=labels)

    metadata = {
        "dataset_kind": "synthetic_baseband_iq",
        "generator_version": "0.1.0",
        "seed": args.seed,
        "sample_rate_hz": args.sample_rate,
        "samples_per_window": args.samples_per_window,
        "sample_format": "complex64",
        "examples_per_class": args.examples,
        "class_counts": {name: int(np.count_nonzero(labels == name)) for name in CLASS_NAMES},
        "classes_are_rogue_labels": False,
        "limitations": "Synthetic examples support pipeline development, not claims of real-world rogue-transmitter detection.",
    }
    sidecar = output.with_suffix(".json")
    sidecar.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(labels)} synthetic IQ windows to {output}")
    print(f"Saved dataset metadata to {sidecar}")


def _train(args: argparse.Namespace) -> None:
    dataset_path = Path(args.dataset)
    with np.load(dataset_path, allow_pickle=False) as dataset:
        iq = dataset["iq"]
        labels = dataset["labels"].astype(str)

    features = np.vstack([extract_features(window) for window in iq])
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels,
    )

    classifier = RandomForestClassifier(
        n_estimators=args.trees,
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=-1,
    )
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)

    print("Synthetic holdout evaluation (signal-pattern classification only)")
    print(classification_report(y_test, predictions, labels=CLASS_NAMES, zero_division=0))
    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y_test, predictions, labels=CLASS_NAMES))

    model_path = Path(args.model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import joblib
    except ImportError as error:  # pragma: no cover - dependency error guidance
        raise SystemExit("Install project dependencies first: python -m pip install -e .") from error
    joblib.dump(
        {
            "classifier": classifier,
            "class_names": CLASS_NAMES,
            "feature_version": "iq-spectrum-v1",
            "training_data_kind": "synthetic_baseband_iq",
        },
        model_path,
    )
    print(f"Saved model to {model_path}")


def _train_cnn(args: argparse.Namespace) -> None:
    train_cnn(
        args.dataset,
        args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_size=args.validation_size,
        max_examples=args.max_examples,
        split_mode=args.split_mode,
        heldout_snrs=args.heldout_snrs,
        patience=args.patience,
        seed=args.seed,
    )


def _inspect_deeprfsoc(args: argparse.Namespace) -> None:
    inspect_deeprfsoc(args.input)


def _import_deeprfsoc(args: argparse.Namespace) -> None:
    import_deeprfsoc(args.input, args.output_dir, seed=args.seed,
                      chunk_windows=args.chunk_windows, split_group_windows=args.split_group_windows)


def _train_deeprfsoc(args: argparse.Namespace) -> None:
    train_deeprfsoc(
        args.dataset_dir, args.model, args.metrics,
        epochs=args.epochs, batch_size=args.batch_size, patience=args.patience,
        seed=args.seed, max_train_per_cell=args.max_train_per_cell,
        max_eval_per_cell=args.max_eval_per_cell,
        resume_from=args.resume_from, latest_checkpoint_path=args.latest_checkpoint,
        device_name=args.device, precision=args.precision,
    )


def _benchmark_deeprfsoc(args: argparse.Namespace) -> None:
    batch_sizes = tuple(int(value.strip()) for value in args.batch_sizes.split(",") if value.strip())
    report = benchmark_deeprfsoc(args.dataset_dir, batches=args.batches, seed=args.seed,
                                 batch_sizes=batch_sizes)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved benchmark: {output}", flush=True)


def _inspect_plkg(args: argparse.Namespace) -> None:
    report = inspect_plkg(args.input_root, output_path=args.output)
    print(json.dumps({
        "dataset": report["dataset"],
        "source_file_count_excluding_git_metadata": report["source_file_count_excluding_git_metadata"],
        "source_bytes_excluding_git_metadata": report["source_bytes_excluding_git_metadata"],
        "pusch_records": report["pusch"]["records"],
        "pdsch_records": report["pdsch"]["records"],
        "pusch_qam_classes": report["pusch"]["qam_class_counts"],
        "pdsch_qam_classes": report["pdsch"]["qam_class_counts"],
        "paired_hdf5": [{"file": Path(item["path"]).name, "pairs": item["pairs"]}
                         for item in report["paired_hdf5"]],
        "inspection_report": str(Path(args.output).resolve()),
        "training_assessment": report["model_training_assessment"]["recommendation"],
    }, indent=2), flush=True)


def _import_plkg(args: argparse.Namespace) -> None:
    result = import_plkg_profile(args.input_root, args.output_dir)
    print(json.dumps(result, indent=2), flush=True)


def _train_plkg_profile(args: argparse.Namespace) -> None:
    train_plkg_profile(
        args.dataset_dir,
        args.model,
        args.metrics,
        seed=args.seed,
        n_estimators=args.n_estimators,
        validation_percentile=args.validation_percentile,
        smoke_max_records_per_split=args.smoke_max_records_per_split,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rogue-detector")
    commands = parser.add_subparsers(dest="command", required=True)

    inspect_plkg_cmd = commands.add_parser("inspect-plkg", help="inspect OAI 5G PLKG raw PUSCH/PDSCH captures")
    inspect_plkg_cmd.add_argument("--input-root", default="data/raw/oai_5g_plkg")
    inspect_plkg_cmd.add_argument("--output", default="outputs/plkg/inspection.json")
    inspect_plkg_cmd.set_defaults(run=_inspect_plkg)

    import_plkg_cmd = commands.add_parser("import-plkg", help="build compact PDSCH profile features from PLKG records")
    import_plkg_cmd.add_argument("--input-root", default="data/raw/oai_5g_plkg")
    import_plkg_cmd.add_argument("--output-dir", default="data/processed/plkg")
    import_plkg_cmd.set_defaults(run=_import_plkg)

    train_plkg_cmd = commands.add_parser("train-plkg-profile", help="fit an exploratory one-class PDSCH profile model")
    train_plkg_cmd.add_argument("--dataset-dir", default="data/processed/plkg")
    train_plkg_cmd.add_argument("--model", default="models/plkg_pdsch_profile.joblib")
    train_plkg_cmd.add_argument("--metrics", default="outputs/plkg/profile_metrics.json")
    train_plkg_cmd.add_argument("--seed", type=int, default=7)
    train_plkg_cmd.add_argument("--n-estimators", type=int, default=250)
    train_plkg_cmd.add_argument("--validation-percentile", type=float, default=95.0)
    train_plkg_cmd.add_argument("--smoke-max-records-per-split", type=int, default=0,
                                help="cap train/validation/test records for a quick smoke run; 0 uses all")
    train_plkg_cmd.set_defaults(run=_train_plkg_profile)

    generate = commands.add_parser("generate", help="generate labelled synthetic IQ windows")
    generate.add_argument("--examples", type=int, default=250, help="examples per class (default: 250)")
    generate.add_argument("--samples-per-window", type=int, default=2048)
    generate.add_argument("--sample-rate", type=float, default=DEFAULT_SAMPLE_RATE_HZ)
    generate.add_argument("--seed", type=int, default=7)
    generate.add_argument("--output", default="data/generated/pilot.npz")
    generate.set_defaults(run=_generate)

    import_data = commands.add_parser("import-radioml", help="convert RadioML2016.10A pickle into project NPZ format")
    import_data.add_argument("--input", required=True, help="path to RML2016.10a_dict.pkl downloaded from Kaggle")
    import_data.add_argument("--output", default="data/processed/radioml2016_10a.npz")
    import_data.add_argument("--modulations", help="optional comma-separated subset, e.g. BPSK,QPSK,8PSK")
    import_data.add_argument("--max-per-class-snr", type=int, default=0, help="limit examples per modulation and SNR cell; 0 keeps all")
    import_data.add_argument("--seed", type=int, default=7)
    import_data.set_defaults(run=_import_radioml)

    import_r16 = commands.add_parser("import-radiomod-r16", help="convert RadioMod-R16 HDF5 into project NPZ format")
    import_r16.add_argument("--input", default="data/raw/RadioMod-R16_dataset.h5")
    import_r16.add_argument("--output", default="data/processed/radiomod_r16.npz")
    import_r16.add_argument("--modulations", default="BPSK,QPSK,8PSK",
                            help="comma-separated subset; defaults to the three classes shared with the RadioML run")
    import_r16.add_argument("--seed", type=int, default=7)
    import_r16.set_defaults(run=_import_radiomod16)

    inspect_deep = commands.add_parser("inspect-deeprfsoc", help="inspect the trusted DeepRFSoC pickle structure")
    inspect_deep.add_argument("--input", default="data/raw/deeprfsoc_rfsoc_quant_amc/deeprfsoc.pkl")
    inspect_deep.set_defaults(run=_inspect_deeprfsoc)

    import_deep = commands.add_parser("import-deeprfsoc", help="memory-map all valid DeepRFSoC windows into processed arrays")
    import_deep.add_argument("--input", default="data/raw/deeprfsoc_rfsoc_quant_amc/deeprfsoc.pkl")
    import_deep.add_argument("--output-dir", default="data/processed/deeprfsoc")
    import_deep.add_argument("--chunk-windows", type=int, default=2048)
    import_deep.add_argument("--split-group-windows", type=int, default=512,
                             help="contiguous windows kept together when making class/SNR-stratified splits")
    import_deep.add_argument("--seed", type=int, default=7)
    import_deep.set_defaults(run=_import_deeprfsoc)

    train_deep = commands.add_parser("train-deeprfsoc", help="train and evaluate the 1D CNN on DeepRFSoC")
    train_deep.add_argument("--dataset-dir", default="data/processed/deeprfsoc")
    train_deep.add_argument("--model", default="models/deeprfsoc_1d_cnn.pt")
    train_deep.add_argument("--metrics", default="outputs/deeprfsoc/metrics.json")
    train_deep.add_argument("--epochs", type=int, default=20)
    train_deep.add_argument("--batch-size", type=int, default=512)
    train_deep.add_argument("--patience", type=int, default=5)
    train_deep.add_argument("--seed", type=int, default=7)
    train_deep.add_argument("--max-train-per-cell", type=int, default=0,
                            help="smoke-test cap per class/SNR cell; 0 uses all training windows")
    train_deep.add_argument("--max-eval-per-cell", type=int, default=0,
                            help="smoke-test cap per class/SNR cell for validation and test; 0 uses all")
    train_deep.add_argument("--resume-from", default=None, help="resume from a latest checkpoint saved by this command")
    train_deep.add_argument("--latest-checkpoint", default="models/deeprfsoc_latest.pt",
                            help="save resumable optimizer state after every epoch")
    train_deep.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    train_deep.add_argument("--precision", choices=("auto", "amp", "fp32"), default="auto",
                            help="auto enables CUDA AMP; benchmark before selecting fp32 for speed")
    train_deep.set_defaults(run=_train_deeprfsoc)

    benchmark_deep = commands.add_parser("benchmark-deeprfsoc", help="benchmark DeepRFSoC memmap-to-CNN throughput")
    benchmark_deep.add_argument("--dataset-dir", default="data/processed/deeprfsoc")
    benchmark_deep.add_argument("--batches", type=int, default=40, help="timed batches per batch-size/precision configuration")
    benchmark_deep.add_argument("--seed", type=int, default=7)
    benchmark_deep.add_argument("--batch-sizes", default="64,128,256,384,512,768,1024,1536,2048",
                                help="comma-separated batch sizes")
    benchmark_deep.add_argument("--output", default="outputs/deeprfsoc/benchmark.json")
    benchmark_deep.set_defaults(run=_benchmark_deeprfsoc)
    benchmark_deep.set_defaults(run=_benchmark_deeprfsoc)

    train_radio = commands.add_parser("train-radioml", help="train/evaluate the feature baseline on imported RadioML")
    train_radio.add_argument("--dataset", default="data/processed/radioml2016_10a.npz")
    train_radio.add_argument("--model", default="models/radioml_feature_baseline.joblib")
    train_radio.add_argument("--test-size", type=float, default=0.2)
    train_radio.add_argument("--trees", type=int, default=200)
    train_radio.add_argument("--seed", type=int, default=7)
    train_radio.add_argument("--max-examples", type=int, default=0, help="randomly cap total data for a quick pilot; 0 uses all")
    train_radio.set_defaults(run=_train_radioml)

    cnn = commands.add_parser("train-cnn", help="train a 1D CNN on RadioML I/Q modulation labels")
    cnn.add_argument("--dataset", default="data/processed/radioml2016_10a.npz")
    cnn.add_argument("--model", default="models/radioml_1d_cnn.pt")
    cnn.add_argument("--epochs", type=int, default=40)
    cnn.add_argument("--batch-size", type=int, default=256)
    cnn.add_argument("--validation-size", type=float, default=0.15,
                     help="fraction of non-held-out SNR data used for validation")
    cnn.add_argument("--max-examples", type=int, default=0, help="randomly cap total windows for a quick run; 0 uses all")
    cnn.add_argument("--split-mode", choices=("per_snr", "snr_holdout"), default="per_snr",
                     help="per_snr trains/tests at every SNR; snr_holdout tests levels excluded from training")
    cnn.add_argument("--heldout-snrs", default="-20,-10,0,10,18",
                     help="comma-separated SNR levels excluded from training and used for final test")
    cnn.add_argument("--patience", type=int, default=8, help="early-stop after this many epochs without validation improvement")
    cnn.add_argument("--seed", type=int, default=7)
    cnn.set_defaults(run=_train_cnn)

    train = commands.add_parser("train", help="train/evaluate the synthetic signal-pattern baseline")
    train.add_argument("--dataset", default="data/generated/pilot.npz")
    train.add_argument("--model", default="models/signal_type_baseline.joblib")
    train.add_argument("--test-size", type=float, default=0.2)
    train.add_argument("--trees", type=int, default=200)
    train.add_argument("--seed", type=int, default=7)
    train.set_defaults(run=_train)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
