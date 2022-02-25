
import jax.scipy as js

from dataclasses import dataclass

class Mixture:
    pass

@dataclass
class Normal(Mixture):
    mean_: str
    std_: str
    mean: int
    std: int
    default_mean: float = 0.0
    default_std: float = 0.001

    def roll(self, u, parameters):
        v = js.stats.norm.ppf(u, parameters[...,self.mean], parameters[...,self.std])
        parameters = parameters.at[...,self.mean].set(v)
        return parameters