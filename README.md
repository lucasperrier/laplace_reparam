# Laplace reparameterization invariance — reproduction

Reproduction of the diagnostic experiments from
[Roy, Miani et al., *Reparameterization invariance in approximate Bayesian inference*, NeurIPS 2024](https://arxiv.org/abs/2406.03334).

The paper's diagnosis: sampled Laplace approximations underfit in
overparametrized neural networks because posterior mass lands in
`ker(GGN)`, a subspace along which the *linearized* predictive is
invariant but the nonlinear network is not.

All code is small JAX/Flax, CPU-only. Each experiment produces a
figure in `figures/`.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m experiments.exp1_basic
python -m experiments.exp2_rank
python -m experiments.exp3_decomposition
```

## What each experiment shows

### Experiment 1 — Sampled vs linearized predictive
`python -m experiments.exp1_basic` → `figures/01_sampled_vs_linearized.png`

Train a 2-layer tanh MLP on 20 points from a sine with a gap, then
compare the sampled and linearized Laplace predictives.

![](figures/01_sampled_vs_linearized.png)

Sampled Laplace (left) has high predictive variance everywhere,
including on the training data. Linearized Laplace (right) tracks
MAP on the training data and widens in the gap and the extrapolation
tails.

### Experiment 2 — Rank(GGN) vs training fit
`python -m experiments.exp2_rank` → `figures/02_rank_vs_underfitting.png`

Sweep `N ∈ {10, …, 1280}`. For each `N`, fit MAP, compute rank(GGN),
and measure the training fit (`1 − MSE/Var(y)`) of MAP, the
linearized-Laplace mean, and the sampled-Laplace mean.

![](figures/02_rank_vs_underfitting.png)

MAP and linearized Laplace sit together at ≈0.98 once there is
enough data. Sampled Laplace stays negative (worse than the
constant-mean baseline) across the entire sweep, including at
`N = 1280 ≫ D = 321`, where the paper's dimensional rank bound
`dim ker(G) ≥ D − NO` is no longer active. The repo does not
identify the mechanism behind this residual underfitting.

### Experiment 3 — Kernel vs image decomposition
`python -m experiments.exp3_decomposition` → `figures/03_kernel_image_decomposition.png`

Eigendecompose the GGN, project posterior samples onto its image and
kernel subspaces, and compute predictives for each projection under
both sampled and linearized Laplace.

![](figures/03_kernel_image_decomposition.png)

Bottom-right panel: the linearized predictive collapses to a thin
band around MAP when samples live entirely in `ker(GGN)`, consistent
with `f_lin(w, x*) = f(w_MAP, x*) + J(x*)(w − w_MAP)` and
`J(x*)v = 0` for `v ∈ ker(GGN)`. The top-right panel — the same
kernel projection evaluated through the *nonlinear* network — shows
large uncertainty. Mean predictive variance from the kernel subspace
in this run: `1.578·10²` (sampled) vs `3.677·10⁻²` (linearized).

## Layout

```
src/
  model.py       MLP, toy data, MAP trainer
  laplace.py     GGN, posterior cov, sampling, predictives
experiments/
  exp1_basic.py
  exp2_rank.py
  exp3_decomposition.py
figures/         output plots
```

## Conventions

Sum-scale Laplace with Gaussian likelihood (σ=1) and Gaussian prior
`N(0, 1/α I)`. Posterior `N(w_MAP, (G + α I)^{-1})` with
`G = J.T @ J` and `J ∈ R^{N×D}` the stacked per-sample Jacobian of
the scalar-output network. Default `α = 0.1`, hidden sizes `(16, 16)`,
`D = 321` parameters. `α` is held fixed across all experiments.

## Scope

This repo reproduces the diagnostic experiments only. Not
implemented:
- The paper's Riemannian Laplace diffusion (the proposed fix).
- Any sweep over the prior precision `α`.
- Any analysis attributing the residual underfitting in Experiment 2
  to a specific mechanism (e.g. architectural symmetries). Such an
  attribution would require additional experiments not included
  here.
