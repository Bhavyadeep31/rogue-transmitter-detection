"""Small, interpretable features for the synthetic signal-type baseline."""

from __future__ import annotations

import numpy as np


SPECTRUM_BINS = 64


def extract_features(iq: np.ndarray) -> np.ndarray:
    """Return time-domain statistics and coarse log-spectrum bins for one IQ window."""
    samples = np.asarray(iq, dtype=np.complex64).reshape(-1)
    if samples.size < SPECTRUM_BINS or not np.isfinite(samples).all():
        raise ValueError(f"IQ window must contain at least {SPECTRUM_BINS} finite samples")

    magnitude = np.abs(samples).astype(np.float64)
    mean_magnitude = float(magnitude.mean())
    magnitude_std = float(magnitude.std())
    rms = float(np.sqrt(np.mean(magnitude**2)))
    peak_to_rms = float(magnitude.max() / max(rms, 1e-12))
    normalized_kurtosis = float(
        np.mean(((magnitude - mean_magnitude) / max(magnitude_std, 1e-12)) ** 4)
    )

    phase_step = np.angle(samples[1:] * np.conjugate(samples[:-1]))
    power = np.abs(np.fft.fftshift(np.fft.fft(samples * np.hanning(samples.size)))) ** 2
    coarse_spectrum = np.array(
        [chunk.mean() for chunk in np.array_split(power, SPECTRUM_BINS)], dtype=np.float64
    )
    coarse_spectrum_db = 10.0 * np.log10(coarse_spectrum + 1e-15)

    scalar_features = np.array(
        [
            np.log10(rms**2 + 1e-15),
            peak_to_rms,
            mean_magnitude,
            magnitude_std,
            normalized_kurtosis,
            float(np.mean(np.cos(phase_step))),
            float(np.mean(np.sin(phase_step))),
            float(np.std(phase_step)),
        ],
        dtype=np.float64,
    )
    return np.concatenate((scalar_features, coarse_spectrum_db))
