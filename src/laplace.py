"""Core Laplace approximation utilities.

Conventions (sum-scale, cf. Roy & Miani et al., NeurIPS 2024):
  * GGN:   G = sum_n J(x_n).T @ J(x_n) = J_stack.T @ J_stack,
    where J_stack has shape (N, D) for scalar output + Gaussian likelihood
    (output-Hessian = I).
  * Laplace posterior: N(w_MAP, (G + alpha * I)^{-1}).
  * Sampled Laplace evaluates f(w, x) for w ~ posterior.
  * Linearized Laplace (eq. 4 of the paper):
        f_lin(w, x*) = f(w_MAP, x*) + J(x*) @ (w - w_MAP)
    which is invariant along kernel(G) by construction.
"""

from typing import Callable, Tuple

import jax
import jax.numpy as jnp


# ---------- Jacobian of the flat-weight network ----------

def _flat_apply(model, unravel: Callable):
    """Return f(w_flat, x) that applies the model from a flat weight vector."""
    def f(w_flat, x):
        return model.apply(unravel(w_flat), x)
    return f


def _jacobian(model, unravel: Callable, w_flat: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """Stacked Jacobian of the scalar-output network at inputs x, shape (N, D)."""
    f = _flat_apply(model, unravel)
    def f_flat(w):
        return f(w, x).reshape(-1)   # (N,)
    return jax.jacrev(f_flat)(w_flat)


# ---------- GGN and posterior ----------

def compute_ggn(model, unravel: Callable, w_flat: jnp.ndarray, x_train: jnp.ndarray) -> jnp.ndarray:
    """Exact dense GGN, shape (D, D). Gaussian likelihood => G = J.T @ J."""
    J = _jacobian(model, unravel, w_flat, x_train)
    return J.T @ J


def compute_posterior_covariance(ggn: jnp.ndarray, alpha: float) -> jnp.ndarray:
    """Return (G + alpha I)^{-1}, symmetrized for numerical stability."""
    D = ggn.shape[0]
    A = 0.5 * (ggn + ggn.T) + alpha * jnp.eye(D)
    return jnp.linalg.inv(A)


def eigendecompose_ggn(ggn: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Symmetric eigendecomposition: eigvals ascending, eigvecs orthonormal."""
    sym = 0.5 * (ggn + ggn.T)
    return jnp.linalg.eigh(sym)


def ggn_rank(eigvals: jnp.ndarray, threshold: float = 1e-6) -> int:
    """Numerical rank = #eigenvalues above `threshold`."""
    return int(jnp.sum(eigvals > threshold))


# ---------- Sampling ----------

def sample_weights(
    w_map: jnp.ndarray,
    cov: jnp.ndarray,
    n_samples: int,
    key: jax.Array,
) -> jnp.ndarray:
    """Draw samples from N(w_MAP, cov). Returns shape (S, D)."""
    D = w_map.shape[0]
    sym = 0.5 * (cov + cov.T)
    # eigendecomposition-based square root: robust when cov is ill-conditioned
    evals, evecs = jnp.linalg.eigh(sym)
    evals = jnp.clip(evals, min=0.0)
    L = evecs * jnp.sqrt(evals)[None, :]   # cov = L @ L.T
    z = jax.random.normal(key, (n_samples, D))
    return w_map[None, :] + z @ L.T


# ---------- Predictives ----------

def predict_sampled(
    model,
    unravel: Callable,
    w_samples: jnp.ndarray,
    x_test: jnp.ndarray,
) -> jnp.ndarray:
    """Nonlinear f(w_i, x*) for each sample. Shape (S, N_test, out_dim)."""
    f = _flat_apply(model, unravel)
    return jax.vmap(lambda w: f(w, x_test))(w_samples)


def predict_linearized(
    model,
    unravel: Callable,
    w_map: jnp.ndarray,
    w_samples: jnp.ndarray,
    x_test: jnp.ndarray,
) -> jnp.ndarray:
    """Linearized predictions (eq. 4): f(w_MAP, x*) + J(x*) (w - w_MAP).

    J(x*) is evaluated once at w_MAP, so kernel(J(x*)) components of
    (w - w_MAP) drop out. Shape (S, N_test, out_dim).
    """
    f = _flat_apply(model, unravel)
    f_map = f(w_map, x_test)                             # (N_test, out_dim)
    J_star = _jacobian(model, unravel, w_map, x_test)    # (N_test, D)
    deltas = w_samples - w_map[None, :]                  # (S, D)
    corrections = deltas @ J_star.T                      # (S, N_test)
    return f_map[None, :, :] + corrections[:, :, None]
