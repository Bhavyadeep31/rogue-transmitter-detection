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

## DeepRFSoC modulation-recognition pipeline

DeepRFSoC is a hardware-loopback modulation dataset. The verified source file in this workspace is `data/raw/deeprfsoc_rfsoc_quant_amc/deeprfsoc.pkl` (2,936,018,580 bytes). Inspection found a dictionary with 112 `(modulation, SNR-string)` keys and 1,433,600 total windows. Each array has shape `(2, 128, N)`, `float64`, where axis 0 is I then Q, axis 1 is the 128-sample sequence, and axis 2 indexes windows. Per-cell window counts vary. Classes are BPSK, CPFSK, GFSK, PAM4, PSK8, QAM16, QAM64, and QPSK; SNR values run from -20 to +30 dB at the dataset's available 4 dB steps, with 28 and 30 dB as the final values. The pickle carries SNR as condition metadata, but no separate channel ID, capture/session ID, center frequency, sample rate, timestamps, or N78 occupancy labels.

The importer converts all finite windows to float32 `.npy` memory maps under `data/processed/deeprfsoc/`, retaining separate labels, SNR and split arrays. The importer must unpickle the official source once; subsequent training reads memory-mapped arrays in batches. It keeps contiguous blocks together within every modulation/SNR cell for a fixed-seed stratified split. As the source has no recording/session IDs, this reduces nearby-window leakage but cannot guarantee session-disjoint evaluation. Training normalizes each IQ window by a shared I/Q RMS value and applies random common phase and circular timing shifts. Best-checkpoint selection prioritizes validation accuracy at SNR <= 0 dB, with validation loss as a tie-breaker.

Run in PowerShell from the repository root. The pickle is an executable serialization format; only import the hash-verified source from the official DeepRFSoC repository.

```powershell
# Verify the CUDA-enabled environment and GPU
.venv-cnn\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

# Inspect exact keys, shapes, dtypes, classes, and SNR cells
.venv-cnn\Scripts\python.exe -m rogue_detector inspect-deeprfsoc --input data/raw/deeprfsoc_rfsoc_quant_amc/deeprfsoc.pkl

# Convert all valid windows; seed and contiguous split-group size are recorded in dataset_metadata.json
.venv-cnn\Scripts\python.exe -u -m rogue_detector import-deeprfsoc --input data/raw/deeprfsoc_rfsoc_quant_amc/deeprfsoc.pkl --output-dir data/processed/deeprfsoc --seed 7 --split-group-windows 512 2>&1 | Tee-Object -FilePath outputs/deeprfsoc/import.log

# Benchmark representative batch sizes and FP32/AMP on the current machine
.venv-cnn\Scripts\python.exe -u -m rogue_detector benchmark-deeprfsoc --dataset-dir data/processed/deeprfsoc --batches 40 --batch-sizes 256,512,1024,1536,2048,3072,4096 --seed 7 --output outputs/deeprfsoc/benchmark.json 2>&1 | Tee-Object -FilePath outputs/deeprfsoc/benchmark.log

# Full training and held-out evaluation; choose --precision based on the local benchmark
.venv-cnn\Scripts\python.exe -u -m rogue_detector train-deeprfsoc --dataset-dir data/processed/deeprfsoc --model models/deeprfsoc_1d_cnn.pt --metrics outputs/deeprfsoc/metrics.json --epochs 20 --batch-size 3072 --patience 5 --seed 7 --device cuda --precision fp32 --latest-checkpoint models/deeprfsoc_latest.pt 2>&1 | Tee-Object -FilePath outputs/deeprfsoc/full_training.log

# Resume at the next epoch from the latest epoch-level optimizer checkpoint
.venv-cnn\Scripts\python.exe -u -m rogue_detector train-deeprfsoc --dataset-dir data/processed/deeprfsoc --model models/deeprfsoc_1d_cnn.pt --metrics outputs/deeprfsoc/metrics.json --epochs 25 --batch-size 3072 --patience 5 --seed 7 --device cuda --precision fp32 --latest-checkpoint models/deeprfsoc_latest.pt --resume-from models/deeprfsoc_latest.pt 2>&1 | Tee-Object -FilePath outputs/deeprfsoc/resume_training.log

# Run the importer, CPU smoke, CUDA smoke, and checkpoint-resume tests
.venv-cnn\Scripts\python.exe -m unittest discover -s tests -v

# Reproduce the same import and training result by rerunning the two commands above with the same source file and seed.
```

The training command evaluates the selected checkpoint on the held-out test split and reports overall accuracy, per-class precision/recall/F1, confusion matrix, and accuracy by SNR. It also writes `metrics.json`, `confusion_matrix.svg`, and `accuracy_by_snr.svg`. A small pipeline smoke run can be performed with `--epochs 1 --batch-size 128 --max-train-per-cell 16 --max-eval-per-cell 8 --device cuda --precision fp32` and distinct smoke output paths. Set `--device cpu` for the CPU smoke path. A `*_latest.pt` file stores the current model, optimizer, scheduler, scaler, random states, best validation state, and epoch after each completed epoch; `--resume-from` continues at the following epoch. `--epochs` is the target epoch number, so increase it when resuming.

The RTX 2050 benchmarks in `outputs/deeprfsoc/benchmark_training_loop.json` and `outputs/deeprfsoc/benchmark_large_batches.json` used the real memory-mapped data and the training loop's normalization, phase/timing augmentation, forward/backward pass, and gradient clipping. The largest measured FP32 throughput was about 18.2k windows/s at batch 3072 (1,280 MiB peak allocated VRAM); batch 4096 was about 18.2k windows/s (1,701 MiB). A separate 20-batch pass found batch 1536 at 16.3k windows/s. CUDA FP32 was more consistent and faster than AMP for the 256–1536 cases in that pass; compare local measurements because laptop clocks and background GPU use affect throughput. The 12-batch large-batch pass included a 24.2k/s AMP result at 2048, but neighboring AMP batch results varied substantially, so it is not used to select the full run. These are short steady-state estimates; full-epoch timing includes validation, startup, disk caching and checkpoint writes. The interrupted CPU baseline log reached 307,200 training windows in about 4.4 minutes at batch 512, corresponding to roughly 1.2k windows/s; it was not a completed epoch. The large-batch benchmark JSON records peak process working set (1.98 GiB) and allocated CUDA memory.

The full RTX 2050 run completed 19 epochs before early stopping (about 20 minutes wall time). Held-out test accuracy was **53.47%** with macro F1 **53.37%**. The weighted test accuracy over SNRs at or below 0 dB was **24.80%**. Accuracy rose from 13.43% at −20 dB to 38.69% at 0 dB, 72.57% at 12 dB, and 88.40% at 30 dB. Per-class F1: BPSK 0.563, CPFSK 0.714, GFSK 0.758, PAM4 0.678, PSK8 0.417, QAM16 0.372, QAM64 0.277, QPSK 0.492. The training accuracy at the stopping epoch was 54.44% and validation accuracy 55.54%, which suggests limited fit rather than a large train/validation overfit gap. The saved RadioML checkpoint scored **70.47%** when evaluated on the reconstructed 6,000-window test split from the 30,000-window, three-class import (`outputs/deeprfsoc/radioml_checkpoint_eval.json`). That is not an apples-to-apples score: RadioML has only 8PSK/BPSK/QPSK and reaches +18 dB, while this DeepRFSoC model has eight classes through +30 dB and its split keeps contiguous 512-window blocks together. Also, the RadioML checkpoint does not record whether its original training used a sample cap, so its recreated split score is indicative. These results still do not demonstrate 5G NR/N78 identification or rogue-transmitter detection.

DeepRFSoC is for modulation classification; it does not label rogue versus legitimate transmitters. Its short labeled IQ windows do not support calibrated clipping/capture-quality checks or 3.3–3.7 GHz band occupancy: those require continuous captures plus sample-rate, center-frequency, ADC-scale, and timing/gap metadata. SNR-stratified classification performance is the available low-SNR evaluation here.

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

### Second benchmark: RadioMod-R16

RadioMod-R16 is an openly licensed, simulated dataset with 16 modulation classes, 512-sample I/Q windows, and impairments including fading, multipath, carrier-frequency offset, and timing drift. It is useful as a second modulation benchmark, but it is not an over-the-air dataset and has no N78 or rogue-transmitter labels. The project importer defaults to the BPSK/QPSK/8PSK subset to match the RadioML experiment.

Download `RadioMod-R16 dataset.h5` from [Zenodo](https://zenodo.org/records/19045423), then run:

```powershell
python -m pip install -e ".[datasets]"
python -m rogue_detector import-radiomod-r16 --input data/raw/RadioMod-R16_dataset.h5 --output data/processed/radiomod_r16_3mod.npz
.venv-cnn\Scripts\python.exe -u -m rogue_detector train-cnn --dataset data/processed/radiomod_r16_3mod.npz --epochs 40 --batch-size 128 --model models/radiomod_r16_1d_cnn.pt
```

On one CPU run with seed 7, test accuracy was 85% overall; accuracy was 40% at -10 dB and 47% at -8 dB, rising to at least 90% from 4 dB onward. This is a small simulated benchmark result, not a direct comparison with RadioML because the datasets use different signal/channel generation and SNR ranges.

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
