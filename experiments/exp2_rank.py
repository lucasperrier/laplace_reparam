"""Experiment 2: rank(GGN) vs. sampled-Laplace training fit.

Reproduces the spirit of Figure 4 in Roy & Miani et al. (NeurIPS 2024):
as N grows the GGN rank grows and the sampled-Laplace predictive mean
improves. Because tanh MLPs have structurally persistent kernel
directions (dead units, symmetries), rank(G) saturates well below D
even when N >> D -- and the sampled predictive mean correspondingly
fails to close the gap with MAP. This mirrors the paper's conclusion:
the underfitting is a symptom of kernel mass, not of finite data.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from src.model import MLP, make_toy_data, train_map
from src import laplace as lap


ALPHA = 0.1
N_SAMPLES = 400
HIDDEN = (16, 16)                     # same as exp1/exp3, D = 321
N_LIST = [10, 20, 40, 80, 160, 320, 640, 1280]
EIG_THRESHOLD = 1e-4


def fit_score(y_true, y_pred):
    """1 - MSE / Var(y): 1 is perfect, 0 is constant-mean baseline."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mse = np.mean((y_true - y_pred) ** 2)
    var = np.var(y_true) + 1e-12
    return 1.0 - mse / var


def run_one(N: int, seed: int):
    x_tr, y_tr, _, _ = make_toy_data(n_train=N, seed=seed)
    model = MLP(hidden_sizes=HIDDEN)
    params, w_map, unravel = train_map(
        model, x_tr, y_tr, alpha=ALPHA, n_steps=5000, seed=seed + 1,
    )
    G = lap.compute_ggn(model, unravel, w_map, x_tr)
    evals, _ = lap.eigendecompose_ggn(G)
    rank = lap.ggn_rank(evals, threshold=EIG_THRESHOLD)

    cov = lap.compute_posterior_covariance(G, alpha=ALPHA)
    key = jax.random.PRNGKey(seed + 777)
    w_samp = lap.sample_weights(w_map, cov, N_SAMPLES, key)

    preds_samp = lap.predict_sampled(model, unravel, w_samp, x_tr)[..., 0]
    preds_lin  = lap.predict_linearized(model, unravel, w_map, w_samp, x_tr)[..., 0]
    y_map_tr = model.apply(params, x_tr)[:, 0]

    return dict(
        N=N, rank=rank, D=w_map.shape[0],
        fit_sampled=fit_score(y_tr, preds_samp.mean(axis=0)),
        fit_lin=fit_score(y_tr, preds_lin.mean(axis=0)),
        fit_map=fit_score(y_tr, y_map_tr),
    )


def main():
    rows = [run_one(N, seed=i) for i, N in enumerate(N_LIST)]
    print("\n  N    rank   fit_MAP   fit_lin   fit_sampled")
    for r in rows:
        print(f"  {r['N']:4d}  {r['rank']:4d}   "
              f"{r['fit_map']:+.3f}    {r['fit_lin']:+.3f}    {r['fit_sampled']:+.3f}")

    ranks = np.array([r["rank"] for r in rows])
    Ns = np.array([r["N"] for r in rows])
    fs = np.array([r["fit_sampled"] for r in rows])
    fl = np.array([r["fit_lin"] for r in rows])
    fm = np.array([r["fit_map"] for r in rows])
    D = rows[0]["D"]

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(ranks, fm, "s--", color="tab:red", label="MAP")
    ax.plot(ranks, fl, "^-", color="tab:green", label="Linearized Laplace (mean)")
    ax.plot(ranks, fs, "o-", color="tab:blue", label="Sampled Laplace (mean)")
    for x, y, N in zip(ranks, fs, Ns):
        ax.annotate(f"N={N}", (x, y), textcoords="offset points", xytext=(4, -12), fontsize=8)
    ax.axhline(1.0, color="gray", lw=0.5, ls=":")
    ax.axhline(0.0, color="gray", lw=0.5, ls=":")
    ax.set_xlabel(f"rank(GGN)   (eigvals > {EIG_THRESHOLD:.0e};  D = {D})")
    ax.set_ylabel(r"training fit   $1 - \mathrm{MSE}/\mathrm{Var}(y)$")
    ax.set_title("Sampled Laplace underfits whenever the GGN kernel carries mass")
    ax.legend(loc="center right")
    fig.tight_layout()
    out = "figures/02_rank_vs_underfitting.png"
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    main()

