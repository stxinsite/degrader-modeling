# degrader-modeling
Mathematical modeling of target protein - degrader- E3 ligase ternary complex binding equilibria. 

Mathematical details and model description can be found [here](https://www.overleaf.com/read/rkzvrfxwdjbg)

## Data
The data used to fit this model come from Nano-BRET assays and can be found [here](https://vantnet.sharepoint.com/:f:/r/sites/SiliconTherapeutics/Dynamite/Shared%20Documents/Biology/2021-08-26_nanobret?csf=1&web=1&e=ZAbI3q). Nano-BRET was used to measure proximity between the VHL E3 ligase and the SMARCA2 target protein. Plates were dosed with varying concentrations of the ACBI1 PROTAC, and mBU was measured as an indication of ternary complex formation. The processed data used to fit this model is in `data/corrected_nanobret_df.csv`. 

## Fitting an MCMC Model 
A NIMBLE model used to run Metropolis-Hastings MCMC sampling and R code for compilation can be found in `equilibrium_nimble.Rmd`. The following constants must be known a priori before fitting the model:
- K_AB: equilibrium dissociation constant of target protein and PROTAC binary complex
- K_BC: equilibrium dissociation constant of E3 ligase and PROTAC binary complex
- alpha_WT: cooperativity of wild type E3 ligase and wild type target protein

The following data must be provided to the model:
- `N`: integer number of data points
- `N_constructs`: integer number of protein mutation constructs *excluding* wild type
- `B_x`: a vector containing extracellular concentrations ([B]_x) of PROTAC 
- `construct_int`: an integer vector containing the cooperativity index corresponding to the data point's construct
- `y`: a vector containing the response, a factor of ternary complex concentration

## MCMC Results
Samples from MCMC chains are saved in the `MCMC` folder and can be read and assigned by
```
samples <- readRDS(file = "MCMC/<RDS filename here>.rds")
```

## Dependencies
### R dependencies
- nleqslv
- nimble
- basicMCMCplots
- coda
- cmdstan
- posterior
- bayesplot
- shinystan
- plyr
- dplyr
- ggplot2
- ggmcmc
- kableExtra
- GGally

### Python dependencies
- numpy
- pandas
- scipy
- lmfit
- pymc3
- theano
- arviz
- seaborn
