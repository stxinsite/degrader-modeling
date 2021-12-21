import arviz as az
import beanmachine.ppl as bm
import pandas as pd
from beanmachine.ppl.inference import VerboseLevel
from torch import tensor

from equilibrium_mcmc_model import EquilibriumModel


if __name__ == '__main__':
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
    FIT
    """
    K_AB = 0.25
    K_BC = 1.8

    construct_index = tensor(nanobret_data['construct'].tolist())
    protac_concentration = tensor(nanobret_data['uM'].tolist())
    mbu = tensor(nanobret_data['mBU_corrected'].tolist())

    em = EquilibriumModel(
        kd_target=K_AB, kd_e3=K_BC,
        protac_ec=protac_concentration,
        construct_idx=construct_index
    )

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

    summary_df = az.summary(samples.to_xarray(), round_to=3)
    print(summary_df.to_string())
    # basic_trace = samples.to_inference_data()
    # az.summary(basic_trace, round_to=3)
    # az.plot_posterior(basic_trace)
