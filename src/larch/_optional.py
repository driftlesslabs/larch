from __future__ import annotations


class Nothing:
    def __getattr__(self, name):
        return Nothing()

    def __call__(self, *args, **kwargs):
        return Nothing()

    def __bool__(self):
        return False


# optional import: JAX
try:
    import jax
    import jax.numpy as jnp
    import jax.scipy as js
except ImportError:
    jax = Nothing()
    jnp = Nothing()
    js = Nothing()
    print("JAX not found. Some functionality will be unavailable.")

# optional import: matplotlib
try:
    from matplotlib import pyplot
except ImportError:
    pyplot = Nothing()


__all__ = [
    "jax",
    "jnp",
    "js",
    "pyplot",
]
