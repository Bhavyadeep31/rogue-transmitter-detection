# Rogue Transmitter Detection

**Project topic:** AI Driven Rogue Transmitter Detection in the 3.3–3.7 GHz 5G NR Band using SDR Spectrum Monitoring.

An educational, receive-only prototype for monitoring selected parts of 5G NR band n78 (3.3–3.8 GHz). The system will capture and describe RF activity, identify signal patterns, and flag unfamiliar activity for human review. A spectrum anomaly alone does not establish that a transmitter is unauthorized or malicious.

## Research topics

1. SDR-based spectrum monitoring architectures for 3GPP N78 (3.3–3.8 GHz): hardware, firmware, sampling and capture chains.
2. Spectrum sensing and anomaly detection for 5G NR waveforms—from energy detection to deep learning.
3. Rogue base station / false base station threat landscape and geolocation of detected emitters.

The Topic 1 design note is in [`docs/lrp-topic-1-sdr-n78.md`](docs/lrp-topic-1-sdr-n78.md).

## First working milestone

The initial software path uses reproducible **synthetic baseband IQ** so data processing and model training can be developed before an SDR is available. It generates four signal-pattern labels (`noise`, `tone`, `bpsk`, `qpsk`), extracts simple IQ/spectrum features, and trains a scikit-learn baseline classifier. These are signal-type labels, not legitimate/rogue labels, and synthetic results must not be presented as real-world detection performance.

### Setup

Use Python 3.10 or newer. From the repository root:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

### Generate and train

```powershell
python -m rogue_detector generate --examples 1000 --output data/generated/pilot.npz
python -m rogue_detector train --dataset data/generated/pilot.npz --model models/signal_type_baseline.joblib
```

Use `--examples 10000` to create a larger balanced dataset. Each example is an independently generated IQ window; the dataset is split by example for the synthetic baseline. Real captures must instead be split by recording session to avoid train/test leakage.

Generated data and trained models are intentionally excluded from Git. The dataset includes a JSON sidecar with sample rate, sample format, class counts, and generator details.

## Modulation recognition with Kaggle RadioML

You can use the [Kaggle RadioML 2016.10A dataset](https://www.kaggle.com/datasets/nolasthitnotomorrow/radioml2016-deepsigcom) as the next benchmark. Download the dataset from its Kaggle page, extract `RML2016.10a_dict.pkl`, and put it at `data/raw/RML2016.10a_dict.pkl`. Review the dataset page and underlying license before redistributing the data or trained artifacts. The file is a Python pickle: only load the file downloaded from the trusted dataset page.

This dataset is for **modulation recognition** across 11 classes and multiple SNR values. It is a research benchmark, not captured specifically in 5G n78, and its labels do not say whether a transmitter is legitimate or rogue. The importer preserves SNR as `snr_db` so model quality can be examined across signal conditions.

```powershell
python -m rogue_detector import-radioml --input data/raw/RML2016.10a_dict.pkl --modulations BPSK,QPSK,8PSK --max-per-class-snr 500
python -m rogue_detector train-radioml --dataset data/processed/radioml2016_10a.npz --max-examples 6000
```

The first command creates a manageable three-modulation pilot. Omit `--modulations` and `--max-per-class-snr` to import the full dataset. `train-radioml` evaluates the Random Forest feature baseline. To train the 1D CNN, install the optional PyTorch dependency and run:

```powershell
# PyTorch on Windows currently supports Python 3.9–3.12. If this project venv
# uses a newer Python, create a separate compatible venv first:
py -3.12 -m venv .venv-cnn
.venv-cnn\\Scripts\\python.exe -m pip install -e ".[deep-learning]"
.venv-cnn\\Scripts\\python.exe -m rogue_detector train-cnn --dataset data/processed/radioml2016_10a.npz --epochs 20 --max-examples 6000
```

The CNN takes raw I/Q as two channels and predicts modulation labels. It normalizes each window by a shared I/Q RMS value, uses phase/timing augmentation, selects the best checkpoint on a validation set, reduces the learning rate when validation stalls, and early-stops when appropriate. By default, training, validation, and test sets each contain examples from every SNR, and the report shows test accuracy by SNR. Use `--split-mode snr_holdout` to instead reserve selected SNR values entirely for a harder unseen-SNR generalization test. This is modulation recognition only; it still does not identify 5G NR, N78, or rogue transmitters.

Run on the full 30,000-window three-class import:

```powershell
.venv-cnn\\Scripts\\python.exe -u -m rogue_detector train-cnn --dataset data/processed/radioml2016_10a.npz --epochs 40 --batch-size 256 --model models/radioml_1d_cnn_snr_eval.pt
```

The SNR holdout is a deliberately harder test: it measures performance at exact SNR levels the model did not train on. Use `--heldout-snrs` to select a different set of held-out SNR values, and keep those values fixed while comparing model changes.

## Planned development path

1. Synthetic IQ generation, feature extraction, and baseline evaluation.
2. Receive-only B210 capture with capture metadata and overflow/gap reporting.
3. Compare synthetic and real controlled captures; add a review/unknown path.
4. Live spectrum/occupancy view and detection history.

Do not transmit test signals over the air for this project without an explicitly approved and lawful test arrangement. For receiver testing, use receive-only ambient captures or a supervisor-approved conducted/shielded setup.

## Scope note

The supplied brief also names “Dataset pipeline for OSCD and LEVIR-CD (or equivalent public datasets)” as a scope item. Those are optical remote-sensing change-detection datasets, so their relationship to the N78 RF project needs confirmation before they are added to this RF data pipeline.
