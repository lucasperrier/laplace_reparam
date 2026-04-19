"""Experiment 1: sampled vs linearized Laplace on the 1D toy problem.

Reproduces Figure 2 (left two panels) of Roy & Miani et al. (NeurIPS 2024):
sampled Laplace visibly underfits (high variance even on training points),
while linearized Laplace fits training points tightly.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from src.model import MLP, make_toy_data, train_map
from src import laplace as lap


ALPHA = 0.1
N_SAMPLES = 100
SEED_SAMPLE = 123


def main():
    # --- data + MAP ---
    x_tr, y_tr, x_te, y_te = make_toy_data(n_train=20, seed=0)
    model = MLP(hidden_sizes=(16, 16))
    _, w_map, unravel = train_map(model, x_tr, y_tr, alpha=ALPHA, n_steps=5000)

    # --- posterior ---
    G = lap.compute_ggn(model, unravel, w_map, x_tr)
    cov = lap.compute_posterior_covariance(G, alpha=ALPHA)
    key = jax.random.PRNGKey(SEED_SAMPLE)
    w_samples = lap.sample_weights(w_map, cov, N_SAMPLES, key)

    # --- predictives ---
    preds_samp = lap.predict_sampled(model, unravel, w_samples, x_te)[..., 0]   # (S, N_te)
    preds_lin  = lap.predict_linearized(model, unravel, w_map, w_samples, x_te)[..., 0]
    y_map = model.apply(unravel(w_map), x_te)[:, 0]

    def mean_std(p):
        return np.asarray(p.mean(axis=0)), np.asarray(p.std(axis=0))

    mu_s, sd_s = mean_std(preds_samp)
    mu_l, sd_l = mean_std(preds_lin)

    # --- plot ---
    xt = np.asarray(x_te[:, 0])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (mu, sd, title) in zip(
        axes,
        [(mu_s, sd_s, "Sampled Laplace"), (mu_l, sd_l, "Linearized Laplace")],
    ):
        ax.fill_between(xt, mu - 2 * sd, mu + 2 * sd, color="tab:blue", alpha=0.25,
                        label=r"$\pm 2\sigma$")
        ax.plot(xt, mu, color="tab:blue", lw=1.6, label="Laplace mean")
        ax.plot(xt, np.asarray(y_map), color="tab:red", lw=1.2, ls="--", label="MAP")
        ax.plot(xt, np.asarray(y_te[:, 0]), color="k", lw=0.8, alpha=0.4, label="truth")
        ax.scatter(np.asarray(x_tr[:, 0]), np.asarray(y_tr[:, 0]),
                   s=18, color="k", zorder=5, label="train")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylim(-2.5, 2.5)
    axes[0].set_ylabel("y")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle(f"Laplace predictive  (D={w_map.shape[0]}, N={x_tr.shape[0]}, "
                 f"rank(G)={lap.ggn_rank(lap.eigendecompose_ggn(G)[0])})")
    fig.tight_layout()
    out = "figures/01_sampled_vs_linearized.png"
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
