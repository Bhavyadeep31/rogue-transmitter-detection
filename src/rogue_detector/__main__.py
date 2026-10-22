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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rogue-detector")
    commands = parser.add_subparsers(dest="command", required=True)

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
