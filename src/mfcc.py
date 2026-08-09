"""DSP front-end selector: pick the MFCC implementation from config.yaml.

Both the trainer (`train_tf.py`) and the quantization calibrator
(`representative_dataset.py`) call `make_frontend(cfg)` to get:

    p, frontend = make_frontend(cfg)

    p        : MFCCParams (frame/mfcc/mel geometry, same for every backend)
    frontend : callable  (batch, n_samples) float32 -> (batch, frames, num_mfcc)
               returned as a plain numpy array.

Backends (config.yaml -> dsp.backend):
    tf       : tf.signal MFCC (original reference; default so existing runs are
               unchanged until you deliberately switch).
    arm_q15  : CMSIS-DSP arm_mfcc_q15 (fixed point, matches on-device arm_math).
    arm_f32  : CMSIS-DSP arm_mfcc_f32 (float parity/debug reference).

Switching backends changes the feature values -> retrain + recompute feature_norm.
"""
from __future__ import annotations

from typing import Callable, Tuple

import numpy as np

from mfcc_tf import MFCCParams


def _dsp_cfg(cfg: dict) -> dict:
    return cfg.get("dsp", {}) or {}


def backend_name(cfg: dict) -> str:
    return str(_dsp_cfg(cfg).get("backend", "tf")).lower()


def make_frontend(cfg: dict) -> Tuple[MFCCParams, Callable[[np.ndarray], np.ndarray]]:
    """Build (MFCCParams, frontend_callable) for the configured DSP backend."""
    p = MFCCParams(cfg)
    backend = backend_name(cfg)
    dsp = _dsp_cfg(cfg)

    if backend == "tf":
        import tensorflow as tf
        from mfcc_tf import waveform_to_mfcc

        def frontend(wavs: np.ndarray) -> np.ndarray:
            wavs = np.asarray(wavs, dtype=np.float32)
            if wavs.ndim == 1:
                wavs = wavs[None, :]
            return waveform_to_mfcc(tf.constant(wavs), p).numpy()

        return p, frontend

    if backend in ("arm_q15", "arm_f32"):
        precision = "q15" if backend == "arm_q15" else "f32"

        # Build the CMSIS-DSP instance lazily, on first actual use. This means a
        # box that trains from a prebuilt mfcc_cache.npz (e.g. Anvil) never needs
        # `cmsisdsp` installed -- it's only required when features are (re)computed.
        state: dict = {}

        def frontend(wavs: np.ndarray) -> np.ndarray:
            if "arm" not in state:
                from mfcc_arm import ArmMFCC
                state["arm"] = ArmMFCC(
                    p,
                    precision=precision,
                    window=str(dsp.get("window", "hann")).lower(),
                    q15_output_scale=float(dsp.get("q15_output_scale", 1.0 / 128.0)),
                )
            return state["arm"].batch(wavs)

        return p, frontend

    raise ValueError(
        f"unknown dsp.backend '{backend}' (use 'tf', 'arm_q15', or 'arm_f32')")
