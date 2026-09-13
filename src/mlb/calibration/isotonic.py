"""Post-hoc probability calibration (isotonic regression), fit ONLY on a
training split and applied to a held-out split — never fit on the same
games it recalibrates, which would just be overfitting dressed up as
calibration.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


def fit_isotonic(train_probs: np.ndarray, train_outcomes: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)
    iso.fit(train_probs, train_outcomes)
    return iso


def apply_isotonic(iso: IsotonicRegression, probs: np.ndarray) -> np.ndarray:
    return iso.predict(probs)
