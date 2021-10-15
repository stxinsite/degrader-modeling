# degrader-modeling
Mathematical modeling of protein degrader ternary complex binding equilibria. 

Mathematical details and model description can be found [here](https://www.overleaf.com/read/rkzvrfxwdjbg)

## Dependencies
### Python dependencies
- numpy
- pandas
- scipy
- lmfit
- pymc3
- theano
- arviz
- seaborn

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

## Fitting an MCMC Model 
A NIMBLE model used to run the Metropolis-Hastings algorithm can be found in `equilibrium_nimble.Rmd`.
The data needed to run the model are described above the `constants` and `data` variables. 