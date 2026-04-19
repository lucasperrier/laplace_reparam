"""Experiment 3: decomposing Laplace variance into kernel vs image of GGN.

Reproduces the right two columns of Figure 2 in Roy & Miani et al.
(NeurIPS 2024). We split the Laplace posterior samples into their
projection onto the image and kernel of the GGN and compute the
predictive variance from each component separately, for both the
sampled (nonlinear) and linearized predictives.

The paper's mechanistic claim (eqs. 5-7):
    * image projections drive predictive variance identically for
      sampled and linearized Laplace;
    * kernel projections have zero effect on linearized Laplace by
      construction, but can move sampled (nonlinear) predictions a lot.
The bottom-right "kernel, linearized" panel should therefore be flat
at zero variance, while "kernel, sampled" is the large variance that
drives sampled-Laplace underfitting.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from src.model import MLP, make_toy_data, train_map
from src import laplace as lap


ALPHA = 0.1
N_SAMPLES = 200
EIG_THRESHOLD = 1e-4
SEED_SAMPLE = 11


def project_samples(w_samples: jnp.ndarray, w_map: jnp.ndarray, U: jnp.ndarray) -> jnp.ndarray:
    """Project (w - w_MAP) onto the subspace spanned by columns of U, then re-center."""
    deltas = w_samples - w_map[None, :]       # (S, D)
    proj   = deltas @ U @ U.T                 # (S, D)
    return w_map[None, :] + proj


def main():
    x_tr, y_tr, x_te, y_te = make_toy_data(n_train=20, seed=0)
    model = MLP(hidden_sizes=(16, 16))
    _, w_map, unravel = train_map(model, x_tr, y_tr, alpha=ALPHA, n_steps=5000)

    G = lap.compute_ggn(model, unravel, w_map, x_tr)
    evals, evecs = lap.eigendecompose_ggn(G)
    image_mask  = evals > EIG_THRESHOLD
    kernel_mask = ~image_mask
    U_img = evecs[:, image_mask]
    U_ker = evecs[:, kernel_mask]
    print(f"D={evecs.shape[0]}  rank(G)={int(image_mask.sum())}  "
          f"nullity(G)={int(kernel_mask.sum())}")

    cov = lap.compute_posterior_covariance(G, alpha=ALPHA)
    key = jax.random.PRNGKey(SEED_SAMPLE)
    w_samples = lap.sample_weights(w_map, cov, N_SAMPLES, key)

    w_img = project_samples(w_samples, w_map, U_img)
    w_ker = project_samples(w_samples, w_map, U_ker)

    # Six predictives: {full, image-only, kernel-only} x {sampled, linearized}
    def run(samples):
        ps = lap.predict_sampled(model, unravel, samples, x_te)[..., 0]
        pl = lap.predict_linearized(model, unravel, w_map, samples, x_te)[..., 0]
        return np.asarray(ps), np.asarray(pl)

    s_full, l_full = run(w_samples)
    s_img,  l_img  = run(w_img)
    s_ker,  l_ker  = run(w_ker)
    y_map = np.asarray(model.apply(unravel(w_map), x_te)[:, 0])

    panels = [
        [("Sampled · full",   s_full), ("Sampled · image",   s_img), ("Sampled · kernel",   s_ker)],
        [("Linearized · full", l_full), ("Linearized · image", l_img), ("Linearized · kernel", l_ker)],
    ]

    xt = np.asarray(x_te[:, 0])
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True, sharey=True)
    for i, row in enumerate(panels):
        for j, (title, p) in enumerate(row):
            mu, sd = p.mean(axis=0), p.std(axis=0)
            ax = axes[i, j]
            ax.fill_between(xt, mu - 2 * sd, mu + 2 * sd,
                            color="tab:blue", alpha=0.25, label=r"$\pm 2\sigma$")
            ax.plot(xt, mu, color="tab:blue", lw=1.4, label="Laplace mean")
            ax.plot(xt, y_map, color="tab:red", lw=1.0, ls="--", label="MAP")
            ax.plot(xt, np.asarray(y_te[:, 0]), color="k", lw=0.6, alpha=0.4, label="truth")
            ax.scatter(np.asarray(x_tr[:, 0]), np.asarray(y_tr[:, 0]),
                       s=14, color="k", zorder=5)
            ax.set_title(title)
            ax.set_ylim(-2.8, 2.8)
            if i == 1:
                ax.set_xlabel("x")
            if j == 0:
                ax.set_ylabel("y")
    axes[0, 0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Laplace predictive decomposed by GGN image/kernel subspaces")
    fig.tight_layout()
    out = "figures/03_kernel_image_decomposition.png"
    fig.savefig(out, dpi=120)
    print(f"saved {out}")

    # numerical sanity: kernel contribution should vanish for linearized
    kernel_var_sampled = jnp.mean(jnp.var(s_ker, axis=0))
    kernel_var_linearized = jnp.mean(jnp.var(l_ker, axis=0))
    print(f"Kernel subspace variance: sampled = {kernel_var_sampled:.3e}, "
          f"linearized = {kernel_var_linearized:.3e}")
    print(f"Ratio: {kernel_var_sampled / kernel_var_linearized:.1e}")


if __name__ == "__main__":
    main()
