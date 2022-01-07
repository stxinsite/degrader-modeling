import numpyro
from numpyro.infer import MCMC, NUTS, Predictive
import numpyro.distributions as dist
from jax import random

from equilibrium_functions import predict_ternary


def model(kd_target: float, kd_e3: float, protac_ec, construct_idx, mbu=None):
    kappa = numpyro.sample("kappa", dist.Uniform(0.0, 1.0))
    total_target = numpyro.sample("total_target", dist.Gamma(0.001, 0.001))

