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
    """SURE-optimal threshold for a vector of (noise-normalized) coefficients
    — MATLAB 'rigrsure'.  risk(k) = (n - 2k + Σ_{i≤k} sx2_i + (n-k)·sx2_k)/n."""
    n = coeffs.size
    if n == 0:
        return 0.0
    sx2 = np.sort(np.abs(coeffs)) ** 2
    cumsum = np.cumsum(sx2)
    i = np.arange(1, n + 1)
    risks = (n - 2 * i + cumsum + (n - i) * sx2) / n
    return float(np.sqrt(sx2[int(np.argmin(risks))]))


def _universal_threshold(n: int) -> float:
    """sqtwolog: t = sqrt(2 ln n) (in noise-normalized units)."""
    return float(np.sqrt(2.0 * np.log(n))) if n > 1 else 0.0


def _minimax_threshold(n: int) -> float:
    """minimaxi: minimax threshold (noise-normalized)."""
    if n <= 32:
        return 0.0
    return float(0.3936 + 0.1829 * np.log2(n))


def _threshold_value(detail: np.ndarray, sigma: float, rule: str) -> float:
    """Physical threshold for one detail band under the chosen MATLAB rule."""
    n = detail.size
    normed = detail / sigma if sigma > 0 else detail
    s = sigma if sigma > 0 else 1.0
    if rule == "rigrsure":
        return _sure_threshold(normed) * s
    if rule == "sqtwolog":
        return _universal_threshold(n) * s
    if rule == "minimaxi":
        return _minimax_threshold(n) * s
    if rule == "heursure":               # choose sure vs universal by energy
        energy = float(np.sum(normed ** 2))
        crit = (np.log2(n) ** 1.5) / np.sqrt(n) if n > 1 else 0.0
        if (energy - n) / n < crit:
            return _universal_threshold(n) * s
        return _sure_threshold(normed) * s
    raise ValueError(f"unknown threshold rule: {rule}")


def wavelet_denoise(signal: np.ndarray, wavelet: str | None = None,
                    level: int | None = None, mode: str | None = None,
                    rule: str | None = None, scale: float | None = None
                    ) -> np.ndarray:
    """Multi-level wavelet thresholding denoiser.

    All defaults come from ``config.DENOISE`` (single source of truth); pass
    explicit args only to override. ``rule`` selects the MATLAB threshold
    selection ('rigrsure' SURE, 'sqtwolog' universal, 'heursure', 'minimaxi');
    ``scale`` multiplies the threshold; the noise std is the MAD of the finest
    detail band. Returns a signal the same length as the input.
    """
    from . import config as C
    d = C.DENOISE
    wavelet = d["wavelet"] if wavelet is None else wavelet
    mode = d["mode"] if mode is None else mode
    rule = d["rule"] if rule is None else rule
    scale = d["scale"] if scale is None else scale
    if level is None:
        level = d.get("level")

    signal = np.array(signal, dtype=float)        # writable copy (pywt needs it)
    n = signal.size
    max_lvl = pywt.dwt_max_level(n, pywt.Wavelet(wavelet).dec_len)
    level = max_lvl if level is None else min(level, max_lvl)

    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745 if coeffs[-1].size else 0.0

    new = [coeffs[0]]
    for c in coeffs[1:]:
        if c.size == 0:
            new.append(c)
            continue
        t = _threshold_value(c, sigma, rule) * scale
        new.append(pywt.threshold(c, t, mode=mode))

    return pywt.waverec(new, wavelet)[:n]


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
