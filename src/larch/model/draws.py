from __future__ import annotations

from .._optional import jax, jnp


def lhs_draw_maker(n_draws, n_cols):
    """
    Create a jitted function that makes draws.

    Parameters
    ----------
    n_draws, n_cols : int

    Returns
    -------
    Callable
        Accepts a jax random state key, returns a new key and the draws
    """

    @jax.jit
    def lhs_draw_column(carry, x):
        key = carry
        k1, k2, k3 = jax.random.split(key, 3)
        return k3, jax.random.permutation(
            k1,
            (jnp.arange(n_draws) + jax.random.uniform(k2, shape=(n_draws,))) / n_draws,
        )

    @jax.jit
    def lhs_draws(key):
        return jax.lax.scan(lhs_draw_column, key, None, length=n_cols)

    return lhs_draws
