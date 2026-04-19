"""Small MLP for 1D regression, trained to MAP (MSE + L2 weight decay)."""

from typing import Tuple
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state
from jax.flatten_util import ravel_pytree


# ---------- Model ----------

class MLP(nn.Module):
    """A plain MLP with tanh activations.

    tanh (rather than ReLU) gives smoother predictives and is the
    classic choice in the BNN literature. Easy to swap later.
    """
    hidden_sizes: Tuple[int, ...] = (16, 16)
    out_dim: int = 1

    @nn.compact
    def __call__(self, x):
        for h in self.hidden_sizes:
            x = nn.Dense(h)(x)
            x = nn.tanh(x)
        x = nn.Dense(self.out_dim)(x)
        return x


# ---------- Data ----------

def make_toy_data(n_train: int = 20, noise_std: float = 0.1, seed: int = 0):
    """Sine curve with a gap in the middle.

    Training x drawn from two clusters on either side of x=0.
    This gap is where Laplace's uncertainty behavior becomes visible.
    """
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)

    # two clusters: [-2, -0.5] and [0.5, 2]
    half = n_train // 2
    x_left = jax.random.uniform(k1, (half,), minval=-2.0, maxval=-0.5)
    x_right = jax.random.uniform(k2, (n_train - half,), minval=0.5, maxval=2.0)
    x_train = jnp.concatenate([x_left, x_right])[:, None]  # shape (N, 1)

    key_noise = jax.random.PRNGKey(seed + 1)
    noise = noise_std * jax.random.normal(key_noise, (n_train, 1))
    y_train = jnp.sin(2.0 * x_train) + noise

    # dense test grid for plotting
    x_test = jnp.linspace(-3.5, 3.5, 400)[:, None]
    y_test = jnp.sin(2.0 * x_test)

    return x_train, y_train, x_test, y_test


# ---------- Training ----------

def loss_fn(params, model, x, y, alpha: float):
    """Negative log-posterior (up to constants) = MSE + L2 weight decay."""
    preds = model.apply(params, x)
    mse = 0.5 * jnp.mean((preds - y) ** 2)
    # alpha is the Gaussian prior precision: p(w) = N(0, 1/alpha)
    flat, _ = ravel_pytree(params)
    reg = 0.5 * alpha * jnp.sum(flat ** 2) / x.shape[0]  # per-sample scale
    return mse + reg


def train_map(
    model: MLP,
    x_train: jnp.ndarray,
    y_train: jnp.ndarray,
    alpha: float = 0.1,
    lr: float = 1e-2,
    n_steps: int = 5000,
    seed: int = 42,
):
    """Train the MLP to a MAP estimate. Returns (params, flat_w_map, unravel_fn)."""
    key = jax.random.PRNGKey(seed)
    params = model.init(key, x_train[:1])  # init with a dummy batch
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, model, x, y, alpha)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    for i in range(n_steps):
        params, opt_state, loss = step(params, opt_state, x_train, y_train)
        if (i + 1) % 1000 == 0:
            print(f"  step {i+1:5d} | loss = {loss:.5f}")

    flat_w_map, unravel_fn = ravel_pytree(params)
    print(f"  final loss = {loss:.5f} | D = {flat_w_map.shape[0]} parameters")
    return params, flat_w_map, unravel_fn


# ---------- Sanity check ----------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x_train, y_train, x_test, y_test = make_toy_data(n_train=20)
    print(f"Training data: {x_train.shape[0]} points")

    model = MLP(hidden_sizes=(16, 16))
    params, flat_w_map, unravel = train_map(model, x_train, y_train, alpha=0.1)

    y_pred = model.apply(params, x_test)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x_test, y_test, "b-", label="true", alpha=0.5)
    ax.plot(x_test, y_pred, "r-", label="MAP prediction")
    ax.scatter(x_train, y_train, c="k", s=20, label="train")
    ax.legend()
    ax.set_title("MAP fit sanity check")
    fig.tight_layout()
    fig.savefig("figures/00_map_fit.png", dpi=120)
    print("Saved figures/00_map_fit.png")