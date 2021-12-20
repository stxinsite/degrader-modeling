
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import beanmachine.ppl as bm
from beanmachine.ppl.inference import VerboseLevel
from beanmachine.ppl.model import RVIdentifier
import torch
from torch import tensor
import torch.distributions as dist
import arviz as az
import equilibrium_functions as eq

"""
NANOBRET DATA
"""
corrected_nanobret_df = pd.read_csv("./data/corrected_nanobret_df.csv")
sorted_minutes = sorted(corrected_nanobret_df['Minutes'].unique().tolist())
subset_minutes = [sorted_minutes[5], sorted_minutes[6]]

# select a subset of data
nanobret_subset = corrected_nanobret_df[corrected_nanobret_df['Minutes'].isin(subset_minutes)].copy()

# subtract minimum mBU within each group from each mBU measurement
nanobret_subset['mBU'] = nanobret_subset.groupby('Construct', group_keys=False).apply(lambda g: g.mBU_corrected - g.mBU_corrected.min())

g = sns.relplot(
    data=nanobret_subset,
    x='uM', y='mBU',
    hue='Construct')
g.set(xscale='log')

construct_dict = {
    'VHL_WT SMARCA2_L1415S': 0,
    'VHL_WT SMARCA2_E1420S': 1,
    'VHL_R69Q SMARCA2_WT': 2,
    'VHL_Y112F SMARCA2_WT': 3,
    'VHL_WT SMARCA2_WT': 4
}
nanobret_subset['construct'] = [construct_dict[item] for item in nanobret_subset['Construct']]

nanobret_data = nanobret_subset[['uM', 'mBU', 'mBU_corrected', 'construct']]

"""
MODEL
"""


class EquilibriumModel(object):
    def __init__(self,
                 kd_target: float,
                 kd_e3: float,
                 protac_ec: torch.Tensor,
                 construct_idx: torch.Tensor):
        self.kd_target = kd_target
        self.kd_e3 = kd_e3
        self.protac_ec = protac_ec
        self.construct_idx = construct_idx
        self.n_construct = construct_idx.unique().size()

    @bm.random_variable
    def kappa(self):
        """Permeability proportion: proportion of PROTAC inside the cell at equilibrium."""
        return dist.Uniform(low=0, high=1)

    @bm.random_variable
    def total_target(self):
        """Total concentration of target protein at equilibrium."""
        return dist.Gamma(concentration=1, rate=1)

    @bm.random_variable
    def total_e3(self):
        """Total concentration of E3 ligase at equilibrium."""
        return dist.Gamma(concentration=1, rate=1)

    @bm.random_variable
    def alpha(self):
        """Cooperativity."""
        return dist.LogNormal(loc=3, scale=0.5).expand(self.n_construct)

    @bm.random_variable
    def beta(self):
        """Scaling factor for ternary complex concentration to mBU."""
        return dist.Normal(loc=0, scale=10)

    @bm.random_variable
    def sigma(self):
        """Standard deviation of predicted mBU."""
        return dist.Gamma(concentration=1, rate=1)

    @bm.functional
    def total_protac(self):
        """Total concentration of intracellular PROTAC at equilibrium."""
        return self.kappa() * self.protac_ec

    @bm.functional
    def wrap_solve_ternary(self, total_protac, alpha):
        total_target = self.total_target().detach()
        total_e3 = self.total_e3().detach()
        return eq.solve_ternary(total_target, total_protac, total_e3, self.kd_target, self.kd_e3, alpha)

    @bm.functional
    def predicted_ternary(self):
        total_protac = self.total_protac()
        alpha = self.alpha()[construct_index]
        predicted_ternary = total_protac.detach().map_(alpha.detach(), self.wrap_solve_ternary)
        return predicted_ternary

    @bm.functional
    def mbu_hat(self):
        return self.predicted_ternary() * torch.exp(self.beta())

    @bm.random_variable
    def predicted_mbu(self):
        """Predicted mBU."""
        return dist.Normal(loc=self.mbu_hat(), scale=self.sigma())


"""
FIT
"""


K_AB = 250e-3
K_BC = 1800e-3

construct_index = tensor(nanobret_data['construct'].tolist())
protac_concentration = tensor(nanobret_data['uM'].tolist())
mbu = tensor(nanobret_data['mBU_corrected'].tolist())

em = EquilibriumModel(kd_target=K_AB, kd_e3=K_BC, protac_ec=protac_concentration, construct_idx=construct_index)

queries = [
    em.kappa(),
    em.total_target(),
    em.total_e3(),
    em.alpha(),
    em.beta(),
    em.sigma(),
]
observations = {em.predicted_mbu(): mbu}

# Run the inference.
samples = bm.GlobalNoUTurnSampler().infer(
    queries=queries,
    observations=observations,
    num_samples=100,
    num_chains=4,
    num_adaptive_samples=0,
    verbose=VerboseLevel.LOAD_BAR,
)

samples.to_xarray()


basic_trace = samples.to_inference_data()
az.summary(basic_trace, round_to=3)
summary_df = az.summary(samples.to_xarray(), round_to=3)
print(summary_df.to_string())

summary_df


az.plot_posterior(basic_trace)
summary_df.head()
summary_df.shape()

