
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import beanmachine.ppl as bm
from beanmachine.ppl.inference import VerboseLevel
from beanmachine.ppl.model import RVIdentifier
import torch
from torch import tensor
import torch.distributions as dist

"""
NANOBRET DATA
"""
corrected_nanobret_df = pd.read_csv("./data/corrected_nanobret_df.csv")
sorted_minutes = sorted(corrected_nanobret_df['Minutes'].unique().tolist())

nanobret_subset = corrected_nanobret_df[corrected_nanobret_df['Minutes'].isin([sorted_minutes[5], sorted_minutes[6]])].copy()
nanobret_subset['mBU_observed'] = nanobret_subset['mBU_corrected'] - nanobret_subset['mBU_corrected'].min()

g = sns.relplot(
    data=nanobret_subset,
    x='uM', y='mBU_observed',
    hue='Construct')
g.set(xscale='log')

construct_dict = {
    'VHL_WT SMARCA2_L1415S': 0,
    'VHL_WT SMARCA2_E1420S': 1,
    'VHL_R69Q SMARCA2_WT': 2,
    'VHL_Y112F SMARCA2_WT': 3,
    'VHL_WT SMARCA2_WT': 4
}
nanobret_subset['Construct'] = [construct_dict[item] for item in nanobret_subset.Construct]
nanobret_subset = nanobret_subset.loc[:,['uM', 'mBU_observed', 'Construct']].copy()

"""
MODEL
"""
K_AB = 250e-3
K_BC = 1800e-3

construct_index = nanobret_subset['Construct'].tolist()
protac_concentration = tensor(nanobret_subset['uM'].tolist())
mbu = tensor(nanobret_subset['mBU_observed'].tolist())

class EquilibriumModel(object):
    def __init__(self, kd_target, kd_e3):
        self.kd_target = kd_target
        self.kd_e3 = kd_e3

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
    def alpha(self, n_construct):
        """Cooperativity."""
        return dist.LogNormal(loc=3, scale=0.5).expand(n_construct)

    @bm.functional
    def alpha_hat(self, construct_idx):
        return self.alpha()[construct_idx]

    @bm.random_variable
    def beta(self):
        """Scaling factor for ternary complex concentration to mBU."""
        return dist.Normal(loc=0, scale=10)

    @bm.random_variable
    def sigma(self):
        """Standard deviation of predicted mBU."""
        return dist.Exponential(rate=10)

    @bm.functional
    def predicted_ternary(self):
        pass

    @bm.random_variable
    def predicted_mbu(self):
        """Predicted mBU."""
        return dist.LogNormal(loc=self.predicted_ternary(), scale=self.sigma())


if __name__ == '__main__':
    print('Imports successful.')