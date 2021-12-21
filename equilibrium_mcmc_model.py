import beanmachine.ppl as bm
import torch
import torch.distributions as dist

import equilibrium_functions as eq


class EquilibriumModel(object):
    """A generative probabilistic model of ternary complex equilibrium.

    Properties created with the ``@property`` decorator should be documented
    in the property's getter method.

    Attributes
    ----------
    kd_target : float
        Binary target protein Kd.
    kd_e3 : float
        Binary E3 ligase Kd.
    protac_ec : torch.Tensor
        Concentrations of extracellular PROTAC.
    construct_idx : torch.Tensor
        Indices for protein construct type.

    """
    def __init__(self,
                 kd_target: float,
                 kd_e3: float,
                 protac_ec: torch.Tensor,
                 construct_idx: torch.Tensor):
        """Docstring for EquilibriumModel.__init__().

        Parameters
        ----------
        kd_target : float
            Binary target protein Kd.
        kd_e3 : float
            Binary E3 ligase Kd.
        protac_ec : torch.Tensor
            Concentrations of extracellular PROTAC.
        construct_idx : torch.Tensor
            Indices for protein construct type.

        """
        self.kd_target = kd_target
        self.kd_e3 = kd_e3
        self.protac_ec = protac_ec
        self.construct_idx = construct_idx
        self.n_construct: torch.Size = construct_idx.unique().size()

    @bm.random_variable
    def kappa(self):
        """Permeability proportion: proportion of PROTAC inside the cell at equilibrium."""
        return dist.Uniform(low=0, high=1)

    @bm.random_variable
    def total_target(self):
        """Total concentration of target protein at equilibrium."""
        return dist.Gamma(concentration=1, rate=1)

    @bm.functional
    def total_protac(self):
        """Total concentration of intracellular PROTAC at equilibrium."""
        return self.kappa() * self.protac_ec

    @bm.random_variable
    def total_e3(self):
        """Total concentration of E3 ligase at equilibrium."""
        return dist.Gamma(concentration=1, rate=1)

    @bm.random_variable
    def alpha(self):
        """Cooperativity."""
        # expand alpha by the number of constructs
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
    def predicted_ternary(self):
        """Predicted equilibrium ternary complex concentrations."""
        total_target = self.total_target()
        total_protac = self.total_protac()
        total_e3 = self.total_e3()
        alpha = self.alpha()[self.construct_idx]

        return eq.solve_equilibrium(
            total_target=total_target,
            total_protac=total_protac,
            total_e3=total_e3,
            kd_target=self.kd_target,
            kd_e3=self.kd_e3,
            alpha=alpha
        )

    @bm.functional
    def mbu_hat(self):
        """Predicted mean mBU."""
        return self.predicted_ternary() * torch.exp(self.beta())

    @bm.random_variable
    def predicted_mbu(self):
        """Predicted mBU."""
        return dist.Normal(loc=self.mbu_hat(), scale=self.sigma())
