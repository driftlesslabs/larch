from __future__ import annotations

from .._optional import jax


def keysplit(key, shapes):
    if key.ndim == 1:
        new_keys = jax.random.split(key, shapes[0])
    else:
        new_keys = jax.vmap(jax.random.split, in_axes=(0, None))(key, shapes[0])
    return new_keys, shapes[1:]
