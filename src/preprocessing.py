"""
Wavelet denoising (paper Sec. 2.2 / 3.1).

The supplied workbook is already wavelet-denoised, so this module is provided
for completeness / re-running the denoising step on raw signals.  It implements
soft thresholding with an adaptive threshold based on Stein's Unbiased Risk
Estimate (SURE), exactly as described in the paper:

    "Wavelet transformation was applied ... using soft thresholding with
     adaptive threshold selection based on Stein's unbiased risk estimate."
"""
from __future__ import annotations

import numpy as np
import pywt


def _sure_threshold(coeffs: np.ndarray) -> float:
    """SURE-optimal threshold for a vector of detail coefficients."""
    n = coeffs.size
    if n == 0:
        return 0.0
    # Standard SURE risk curve (matches MATLAB 'rigrsure').  For sorted ascending
    # squared coefficients sx2 and candidate threshold t^2 = sx2[k]:
    #   risk(k) = (n - 2k + sum_{i<=k} sx2_i + (n-k)*sx2_k) / n
    sx2 = np.sort(np.abs(coeffs)) ** 2
    cumsum = np.cumsum(sx2)
    i = np.arange(1, n + 1)
    risks = (n - 2 * i + cumsum + (n - i) * sx2) / n
    best = int(np.argmin(risks))
    return float(np.sqrt(sx2[best]))


def wavelet_denoise(signal: np.ndarray, wavelet: str = "sym8",
                    level: int | None = None, mode: str = "soft") -> np.ndarray:
    """Denoise a 1-D signal with multi-level wavelet soft thresholding.

    Uses a SURE-adaptive threshold scaled by a robust noise estimate (median
    absolute deviation of the finest detail band).  Returns a signal of the
    same length as the input.
    """
    signal = np.array(signal, dtype=float)        # writable copy (pywt needs it)
    n = signal.size
    if level is None:
        level = min(pywt.dwt_max_level(n, pywt.Wavelet(wavelet).dec_len), 6)

    coeffs = pywt.wavedec(signal, wavelet, level=level)
    # Robust noise std from finest details.
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745 if coeffs[-1].size else 0.0

    new = [coeffs[0]]
    for c in coeffs[1:]:
        if c.size == 0:
            new.append(c)
            continue
        normed = c / sigma if sigma > 0 else c
        t = _sure_threshold(normed) * (sigma if sigma > 0 else 1.0)
        new.append(pywt.threshold(c, t, mode=mode))

    rec = pywt.waverec(new, wavelet)
    return rec[:n]


def wavelet_approx(signal: np.ndarray, wavelet: str = "db4",
                   level: int = 4) -> np.ndarray:
    """Heavy low-pass denoising: keep only the level-``level`` approximation
    band and discard ALL detail bands.

    This reproduces the aggressive smoothing regime of the supplied
    ``NM..NoiseFree`` files (lag-1 autocorrelation ≈ 0.9997 — far smoother than
    soft thresholding, which preserves transients). It is equivalent to a
    ~2**level-day low-pass and is provided to reconstruct that earlier
    (Dec-2023) processing iteration, distinct from the lighter soft-threshold
    denoising used for the final paper dataset.
    """
    signal = np.array(signal, dtype=float)
    n = signal.size
    level = min(level, pywt.dwt_max_level(n, pywt.Wavelet(wavelet).dec_len))
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    coeffs = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet)[:n]


def denoise(signal, method: str = "soft", **kwargs):
    """Unified entry point.

    * ``method='soft'``  -> SURE soft-threshold (paper / final SD dataset; light,
      preserves spikes).
    * ``method='approx'`` -> approximation-only low-pass (NoiseFree files; heavy).
    """
    if method == "soft":
        return wavelet_denoise(signal, **kwargs)
    if method == "approx":
        return wavelet_approx(signal, **kwargs)
    raise ValueError(f"unknown denoise method: {method}")
