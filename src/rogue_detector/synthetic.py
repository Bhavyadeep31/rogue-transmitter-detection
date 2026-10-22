"""Parameterized synthetic complex-IQ examples for pipeline development."""

from __future__ import annotations

import numpy as np


CLASS_NAMES = ("noise", "tone", "bpsk", "qpsk")
DEFAULT_SAMPLE_RATE_HZ = 1_000_000.0


def _complex_noise(rng: np.random.Generator, n_samples: int, std: float) -> np.ndarray:
    return (
        rng.normal(0.0, std / np.sqrt(2.0), n_samples)
        + 1j * rng.normal(0.0, std / np.sqrt(2.0), n_samples)
    )


def _one_window(
    rng: np.random.Generator,
    label: str,
    n_samples: int,
    sample_rate_hz: float,
) -> np.ndarray:
    if label == "noise":
        noise_floor_dbfs = rng.uniform(-38.0, -24.0)
        noise_std = 10.0 ** (noise_floor_dbfs / 20.0)
        noise = _complex_noise(rng, n_samples, noise_std)
        return noise.astype(np.complex64)

    amplitude = rng.uniform(0.15, 0.8)
    frequency_offset_hz = rng.uniform(-0.28, 0.28) * sample_rate_hz
    phase = rng.uniform(-np.pi, np.pi)
    sample_index = np.arange(n_samples, dtype=np.float64)
    carrier = np.exp(
        1j * (2.0 * np.pi * frequency_offset_hz / sample_rate_hz * sample_index + phase)
    )

    if label == "tone":
        snr_db = rng.uniform(-2.0, 28.0)
        signal = amplitude * carrier
        local_noise = _complex_noise(rng, n_samples, amplitude * 10.0 ** (-snr_db / 20.0))
        return (signal + local_noise).astype(np.complex64)

    if label in ("bpsk", "qpsk"):
        samples_per_symbol = 8
        if n_samples % samples_per_symbol:
            raise ValueError("samples_per_window must be divisible by 8 for digital examples")
        n_symbols = n_samples // samples_per_symbol
        if label == "bpsk":
            symbols = rng.choice(np.array([-1.0, 1.0]), size=n_symbols).astype(np.complex128)
        else:
            phase_index = rng.integers(0, 4, size=n_symbols)
            symbols = np.exp(1j * (np.pi / 4.0 + phase_index * np.pi / 2.0))

        baseband = np.repeat(symbols, samples_per_symbol)
        snr_db = rng.uniform(-2.0, 28.0)
        signal = amplitude * baseband * carrier
        local_noise = _complex_noise(rng, n_samples, amplitude * 10.0 ** (-snr_db / 20.0))
        # Modest timing variation makes independent windows less identical.
        shift = int(rng.integers(0, samples_per_symbol))
        return np.roll(signal + local_noise, shift).astype(np.complex64)

    raise ValueError(f"Unsupported synthetic class: {label}")


def generate_dataset(
    examples_per_class: int = 250,
    n_samples: int = 2048,
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a balanced set of independently randomized synthetic IQ windows."""
    if examples_per_class < 1:
        raise ValueError("examples_per_class must be positive")
    if n_samples < 64 or n_samples % 8:
        raise ValueError("n_samples must be at least 64 and divisible by 8")
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")

    rng = np.random.default_rng(seed)
    total = examples_per_class * len(CLASS_NAMES)
    iq = np.empty((total, n_samples), dtype=np.complex64)
    labels = np.empty(total, dtype="U8")
    position = 0

    for label in CLASS_NAMES:
        for _ in range(examples_per_class):
            iq[position] = _one_window(rng, label, n_samples, sample_rate_hz)
            labels[position] = label
            position += 1

    order = rng.permutation(total)
    return iq[order], labels[order]
