#!/usr/bin/env python3
"""Cached SVD Ridge for audio-only CV with an unpenalized intercept."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np


class AudioRidgeError(RuntimeError):
    pass


@dataclass
class AudioRidgeFold:
    mean: np.ndarray
    scale: np.ndarray
    u: np.ndarray
    singular_values: np.ndarray
    vt: np.ndarray
    validation_basis: np.ndarray
    n_train: int

    def predict_all(self, y_train: np.ndarray,
                    alphas: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        y = np.asarray(y_train, dtype=np.float64)
        alpha = np.asarray(alphas, dtype=np.float64)
        if y.ndim == 1:
            y = y[:, None]
        if y.ndim != 2 or y.shape[0] != self.n_train or not np.isfinite(y).all():
            raise AudioRidgeError("training targets must be finite and match training rows")
        if alpha.ndim != 1 or alpha.size == 0 or np.any(alpha <= 0) or not np.isfinite(alpha).all():
            raise AudioRidgeError("alphas must be a nonempty finite positive vector")
        target_mean = y.mean(axis=0, keepdims=True)
        centered = y - target_mean
        projected = self.u.T @ centered
        factors = self.singular_values[:, None] / (
            self.singular_values[:, None] ** 2 + alpha[None, :]
        )
        audio = np.einsum("vr,ra,rt->vat", self.validation_basis, factors,
                          projected, optimize=True)
        return audio + target_mean[None, :, :], target_mean.ravel()


def prepare_audio_fold(x_train: np.ndarray, x_validation: np.ndarray) -> AudioRidgeFold:
    x_train = np.asarray(x_train, dtype=np.float64)
    x_validation = np.asarray(x_validation, dtype=np.float64)
    if (x_train.ndim != 2 or x_validation.ndim != 2 or
            x_train.shape[1] != x_validation.shape[1] or x_train.shape[0] == 0):
        raise AudioRidgeError("audio fold shapes are invalid")
    if not np.isfinite(x_train).all() or not np.isfinite(x_validation).all():
        raise AudioRidgeError("audio fold contains NaN or Inf")
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale == 0] = 1.0
    train = (x_train - mean) / scale
    validation = (x_validation - mean) / scale
    u, singular_values, vt = np.linalg.svd(train, full_matrices=False)
    return AudioRidgeFold(mean, scale, u, singular_values, vt,
                          validation @ vt.T, len(x_train))
