# Data layout and handling

```text
data/
├── generated/   # Reproducible synthetic IQ datasets; ignored by Git
├── processed/   # Derived windows/features; ignored by Git
└── raw/         # Original SDR captures and sidecars; ignored by Git
```

Keep original captures immutable. For each capture, record at least: label and how it was established, capture/session ID, UTC timestamp, center frequency, complex sample rate, gain/AGC mode, antenna and RF path, IQ format, duration, SDR/driver/software versions, and overflow or timestamp-gap events. Do not put private or sensitive captured content in the repository.

The synthetic generator writes a compressed `.npz` IQ dataset plus a `.json` sidecar. Synthetic signal classes are examples for pipeline development; they are not ground truth for “rogue” transmitters.
