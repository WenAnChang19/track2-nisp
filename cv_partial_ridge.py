#!/usr/bin/env python3
"""Numerical core for CV with unpenalized demographics and penalized audio."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np


class CVError(RuntimeError):
    """Raised when a CV numerical or protocol invariant is violated."""


@dataclass
class PartialFold:
    """Training-only transforms and one reusable SVD for a fold's audio matrix."""

    c_train: np.ndarray
    c_validation: np.ndarray
    c_pinv: np.ndarray
    z_train: np.ndarray
    z_validation: np.ndarray
    u: np.ndarray
    singular_values: np.ndarray
    vt: np.ndarray
    validation_basis: np.ndarray
    mean: np.ndarray
    scale: np.ndarray

    def predict_all(self, y_train: np.ndarray, alphas: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        y = np.asarray(y_train, dtype=np.float64)
        alpha_array = np.asarray(alphas, dtype=np.float64)
        if alpha_array.ndim != 1 or alpha_array.size == 0 or not np.isfinite(alpha_array).all() or np.any(alpha_array <= 0):
            raise CVError("alphas must be a nonempty finite positive vector")
        if y.ndim == 1:
            y = y[:, None]
        if y.ndim != 2 or y.shape[0] != self.c_train.shape[0] or not np.isfinite(y).all():
            raise CVError("training target shape or values are invalid")
        a = self.c_pinv @ y
        residual = y - self.c_train @ a
        projected = self.u.T @ residual
        denominators = self.singular_values[:, None] ** 2 + alpha_array[None, :]
        factors = self.singular_values[:, None] / denominators
        # [validation, alpha, target-column]
        audio = np.einsum(
            "vr,ra,rt->vat",
            self.validation_basis,
            factors,
            projected,
            optimize=True,
        )
        baseline = self.c_validation @ a
        predictions = audio + baseline[:, None, :]
        return predictions, baseline


def prepare_partial_fold(c_train: np.ndarray, c_validation: np.ndarray,
                         x_train: np.ndarray, x_validation: np.ndarray) -> PartialFold:
    c_train = np.asarray(c_train, dtype=np.float64)
    c_validation = np.asarray(c_validation, dtype=np.float64)
    x_train = np.asarray(x_train, dtype=np.float64)
    x_validation = np.asarray(x_validation, dtype=np.float64)
    if c_train.ndim != 2 or c_train.shape[1] != 3:
        raise CVError("demographic design must have intercept, age and sex")
    if c_validation.ndim != 2 or c_validation.shape[1] != 3:
        raise CVError("validation demographic design must have three columns")
    if np.linalg.matrix_rank(c_train) < 3:
        raise CVError("training demographic design rank is less than 3")
    if (x_train.ndim != 2 or x_validation.ndim != 2 or
            x_train.shape[1] != x_validation.shape[1]):
        raise CVError("audio feature shapes differ")
    if c_train.shape[0] != x_train.shape[0] or c_validation.shape[0] != x_validation.shape[0]:
        raise CVError("demographic and audio row counts differ")
    if not all(np.isfinite(value).all()
               for value in (c_train, c_validation, x_train, x_validation)):
        raise CVError("fold inputs contain NaN or Inf")
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale == 0] = 1.0
    x_train_scaled = (x_train - mean) / scale
    x_validation_scaled = (x_validation - mean) / scale
    c_pinv = np.linalg.pinv(c_train)
    b = c_pinv @ x_train_scaled
    z_train = x_train_scaled - c_train @ b
    z_validation = x_validation_scaled - c_validation @ b
    u, singular_values, vt = np.linalg.svd(z_train, full_matrices=False)
    validation_basis = z_validation @ vt.T
    return PartialFold(c_train, c_validation, c_pinv, z_train, z_validation,
                       u, singular_values, vt, validation_basis, mean, scale)


def direct_partial_penalty_prediction(c_train: np.ndarray, c_validation: np.ndarray,
                                      x_train: np.ndarray, x_validation: np.ndarray,
                                      y_train: np.ndarray, alpha: float) -> np.ndarray:
    """Independent oracle that applies training-fold scaling then an augmented solve."""
    c_train = np.asarray(c_train, dtype=np.float64)
    x_train = np.asarray(x_train, dtype=np.float64)
    x_validation = np.asarray(x_validation, dtype=np.float64)
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale == 0] = 1.0
    x_train = (x_train - mean) / scale
    x_validation = (x_validation - mean) / scale
    width = x_train.shape[1]
    design = np.concatenate((c_train, x_train), axis=1)
    penalty = np.concatenate((np.zeros((width, c_train.shape[1])),
                              np.sqrt(alpha) * np.eye(width)), axis=1)
    augmented = np.concatenate((design, penalty), axis=0)
    target = np.concatenate((np.asarray(y_train, dtype=np.float64), np.zeros(width)))
    coefficients = np.linalg.lstsq(augmented, target, rcond=None)[0]
    return np.concatenate((c_validation, x_validation), axis=1) @ coefficients


def choose_alpha(scores: Mapping[float, float], tolerance: float) -> float:
    if not scores:
        raise CVError("alpha scores are empty")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise CVError("alpha tie tolerance must be finite and nonnegative")
    if any(not np.isfinite(alpha) or alpha <= 0 or not np.isfinite(score)
           for alpha, score in scores.items()):
        raise CVError("alpha scores contain invalid values")
    minimum = min(scores.values())
    return float(max(alpha for alpha, score in scores.items() if score <= minimum + tolerance))


def stratified_permutation(y: np.ndarray, groups: np.ndarray,
                           rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, str]:
    """Shuffle target within each group and return the explicit source mapping."""
    y = np.asarray(y, dtype=np.float64)
    groups = np.asarray(groups)
    if (y.ndim != 1 or groups.ndim != 1 or len(y) != len(groups) or len(y) == 0 or
            not np.isfinite(y).all()):
        raise CVError("permutation target and groups must be same-length nonempty vectors")
    mapping = np.arange(len(y), dtype=np.int64)
    for group in np.unique(groups):
        positions = np.flatnonzero(groups == group)
        mapping[positions] = rng.permutation(positions)
    permuted = y[mapping]
    digest = hashlib.sha256(
        json.dumps(mapping.tolist(), separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return permuted, mapping, digest


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if (actual.ndim != 1 or predicted.ndim != 1 or actual.shape != predicted.shape or
            actual.size == 0 or not np.isfinite(actual).all() or not np.isfinite(predicted).all()):
        raise CVError("metric arrays must be same-shape nonempty finite vectors")
    residual = actual - predicted
    denominator = np.sum((actual - actual.mean()) ** 2)
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "r2": float(1.0 - np.sum(residual ** 2) / denominator) if denominator > 0 else 0.0,
        "n": int(len(actual)),
    }
