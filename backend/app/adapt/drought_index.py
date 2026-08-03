from __future__ import annotations

import numpy as np
from scipy.stats import norm
from scipy.stats import kendalltau

DEFAULT_EPS = 1e-6


def _clip_unit_interval(p: np.ndarray, eps: float = DEFAULT_EPS) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    return np.clip(p, eps, 1.0 - eps)


def _as_2d_array(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=np.float64)
    if u.ndim == 1:
        u = u.reshape(-1, 1)
    if u.ndim != 2:
        raise ValueError(f"Expected a 1D or 2D array of pseudo-observations, got shape {u.shape}")
    if u.shape[0] == 0:
        raise ValueError("At least one sample is required.")
    return u


def _clayton_joint_cdf(u: np.ndarray, tau: float = 0.4, eps: float = DEFAULT_EPS) -> np.ndarray:
    u = _as_2d_array(u)
    u = _clip_unit_interval(u, eps=eps)
    d = u.shape[1]

    if d == 1:
        return u[:, 0].astype(np.float32)

    theta = _clayton_theta_from_tau(tau)
    inner = np.sum(np.power(u, -theta), axis=1) - float(d) + 1.0
    inner = np.maximum(inner, 1e-12)
    cdf = np.power(inner, -1.0 / theta)
    return _clip_unit_interval(cdf, eps=eps).astype(np.float32)


def _clayton_kendall_cdf(t: np.ndarray, tau: float = 0.4, eps: float = DEFAULT_EPS) -> np.ndarray:
    """computes K_C(t) = P(C(U) <= t) for a bivariate Clayton copula"""
    t = _clip_unit_interval(t, eps=eps)
    theta = _clayton_theta_from_tau(tau)
    k = t * (1.0 + 1.0 / theta) - np.power(t, theta + 1.0) / theta
    return _clip_unit_interval(k, eps=eps).astype(np.float32)


class ClaytonKendallStandardizedIndex:
    """standardizes a Clayton copula with empirical Kendall correction"""

    def __init__(self, tau: float = 0.4, eps: float = DEFAULT_EPS):
        self.tau = float(tau)
        self.eps = float(eps)
        self.d_: int | None = None
        self.n_train_: int = 0
        self.sorted_t_train_: np.ndarray | None = None

    def fit(self, u_train: np.ndarray) -> "ClaytonKendallStandardizedIndex":
        u_train = _as_2d_array(u_train)
        self.d_ = int(u_train.shape[1])
        t_train = _clayton_joint_cdf(u_train, tau=self.tau, eps=self.eps)
        return self.fit_from_joint_cdf(t_train, d=self.d_)

    def fit_from_joint_cdf(
        self,
        t_train: np.ndarray,
        *,
        d: int | None = None,
    ) -> "ClaytonKendallStandardizedIndex":
        t_train = np.asarray(t_train, dtype=np.float64).ravel()
        t_train = t_train[np.isfinite(t_train)]
        if t_train.size == 0:
            raise ValueError("t_train must contain at least one finite training joint CDF value.")

        self.d_ = int(d) if d is not None else self.d_
        self.sorted_t_train_ = np.sort(_clip_unit_interval(t_train, eps=self.eps))
        self.n_train_ = int(self.sorted_t_train_.size)
        return self

    def _check_fitted(self) -> None:
        if self.sorted_t_train_ is None or self.n_train_ == 0:
            raise RuntimeError("fit() must be called before transform().")

    def _kendall_cdf(self, t: np.ndarray) -> np.ndarray:
        """evaluates empirical K_C(t) = P(T <= t)"""
        self._check_fitted()

        t = np.asarray(t, dtype=np.float64)
        t_shape = t.shape
        t_flat = _clip_unit_interval(t.ravel(), eps=self.eps)

        if self.d_ == 1:
            return t_flat.reshape(t_shape).astype(np.float32)

        count = np.searchsorted(self.sorted_t_train_, t_flat, side="right")
        k = count / (self.n_train_ + 1.0)
        k = _clip_unit_interval(k, eps=self.eps)
        return k.reshape(t_shape).astype(np.float32)

    def transform(self, u: np.ndarray) -> np.ndarray:
        self._check_fitted()
        u = _as_2d_array(u)
        if self.d_ is not None and u.shape[1] != self.d_:
            raise ValueError(f"Expected {self.d_} features but got {u.shape[1]}.")
        t = _clayton_joint_cdf(u, tau=self.tau, eps=self.eps)
        return self.transform_from_joint_cdf(t)

    def transform_from_joint_cdf(self, t: np.ndarray) -> np.ndarray:
        k = self._kendall_cdf(t)
        index = norm.ppf(_clip_unit_interval(k, eps=self.eps))
        if not np.all(np.isfinite(index)):
            raise ValueError("Kendall-corrected standardized index contains non-finite values.")
        return index.astype(np.float32)

    def transform_naive(self, u: np.ndarray) -> np.ndarray:
        self._check_fitted()
        u = _as_2d_array(u)
        if self.d_ is not None and u.shape[1] != self.d_:
            raise ValueError(f"Expected {self.d_} features but got {u.shape[1]}.")
        t = _clayton_joint_cdf(u, tau=self.tau, eps=self.eps)
        return self.transform_naive_from_joint_cdf(t)

    def transform_naive_from_joint_cdf(self, t: np.ndarray) -> np.ndarray:
        t = _clip_unit_interval(t, eps=self.eps)
        index = norm.ppf(t)
        if not np.all(np.isfinite(index)):
            raise ValueError("Naive standardized index contains non-finite values.")
        return index.astype(np.float32)


def z_to_u(z: np.ndarray) -> np.ndarray:
    return norm.cdf(z)


def _clayton_theta_from_tau(tau: float) -> float:
    tau = float(np.clip(tau, 1e-6, 0.999))
    return 2.0 * tau / (1.0 - tau)


def clayton_lower_tail_joint(u: np.ndarray, v: np.ndarray, tau: float = 0.4) -> np.ndarray:
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    u = _clip_unit_interval(u, eps=DEFAULT_EPS)
    v = _clip_unit_interval(v, eps=DEFAULT_EPS)

    theta = _clayton_theta_from_tau(tau)

    inner = np.power(u, -theta) + np.power(v, -theta) - 1.0
    inner = np.maximum(inner, 1e-12)
    C = np.power(inner, -1.0 / theta)
    return C.astype(np.float32)


def msdi_kendall_from_zscores(
    z_et: np.ndarray,
    z_sm: np.ndarray,
    tau: float = 0.4,
) -> np.ndarray:
    """computes the bivariate ET-SM MSDI-Kendall index"""
    z_et = np.asarray(z_et)
    z_sm = np.asarray(z_sm)
    if z_et.shape != z_sm.shape:
        raise ValueError(f"Expected matching z-score shapes, got ET={z_et.shape} and SM={z_sm.shape}.")
    valid = np.isfinite(z_et) & np.isfinite(z_sm)
    index = np.full(z_et.shape, np.nan, dtype=np.float32)
    if not np.any(valid):
        return index

    u = np.column_stack(
        [
            np.asarray(z_to_u(z_et[valid]), dtype=np.float64).ravel(),
            np.asarray(z_to_u(z_sm[valid]), dtype=np.float64).ravel(),
        ]
    )
    t = _clayton_joint_cdf(u, tau=tau, eps=DEFAULT_EPS)
    k = _clayton_kendall_cdf(t, tau=tau, eps=DEFAULT_EPS)
    valid_index = norm.ppf(_clip_unit_interval(k, eps=DEFAULT_EPS))
    if not np.all(np.isfinite(valid_index)):
        raise ValueError("MSDI-Kendall standardized index contains non-finite values.")
    index[valid] = valid_index.astype(np.float32)
    return index


def estimate_tau_from_history(u_hist: np.ndarray, v_hist: np.ndarray) -> float:
    u = np.asarray(u_hist).ravel()
    v = np.asarray(v_hist).ravel()
    m = np.isfinite(u) & np.isfinite(v)
    if m.sum() < 10:
        return 0.4
    tau, _ = kendalltau(u[m], v[m])
    if not np.isfinite(tau):
        return 0.4
    return float(np.clip(tau, 0.05, 0.95))


def _apply_sensitivity(pdry: np.ndarray, sensitivity: float = 1.0) -> np.ndarray:
    p = np.asarray(pdry, dtype=np.float32)
    s = float(np.clip(sensitivity, 0.5, 2.0))
    return np.clip(1.0 - np.power(1.0 - p, s), 0.0, 1.0).astype(np.float32)


def to_usdm_category(pdry: np.ndarray, sensitivity: float = 1.0) -> np.ndarray:
    """maps dryness severity to categories 0-5"""
    p = _apply_sensitivity(pdry, sensitivity=sensitivity)
    cat = np.zeros_like(p, dtype=np.int8)

    cat[p >= 0.70] = 1
    cat[p >= 0.80] = 2
    cat[p >= 0.90] = 3
    cat[p >= 0.95] = 4
    cat[p >= 0.98] = 5
    return cat


def drought_from_zscores(
    z_et: np.ndarray,
    z_sm: np.ndarray,
    tau: float = 0.4,
    sensitivity: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """converts ET-SM z-scores to dryness severity and category"""
    msdi_k = msdi_kendall_from_zscores(z_et=z_et, z_sm=z_sm, tau=tau)
    pdry = norm.cdf(-np.asarray(msdi_k, dtype=np.float64)).astype(np.float32)

    pdry_sensitive = _apply_sensitivity(pdry, sensitivity=sensitivity)
    cat = to_usdm_category(pdry_sensitive, sensitivity=1.0)
    return pdry_sensitive, cat
