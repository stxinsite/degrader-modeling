library(nleqslv)
library(nimble)
library(basicMCMCplots)
library(coda)
library(dplyr)
library(ggplot2)

equilibrium_f <- function(x, parms) {
  A_t <- parms[1]
  B_t <- parms[2]
  C_t <- parms[3]
  K_AB <- parms[4]
  K_BC <- parms[5]
  alpha <- parms[6]

  A <- exp(x[1])
  C <- exp(x[2])
  ABC <- exp(x[3])

  F1 <- A + K_BC * ABC / (alpha * C) + ABC - A_t
  F2 <- K_AB * K_BC * ABC / (alpha * A * C) + K_BC * ABC / (alpha * C) + K_AB * ABC / (alpha * A) + ABC - B_t
  F3 <- C + K_AB * ABC / (alpha * A) + ABC - C_t
  return(c(F1 = F1, F2 = F2, F3 = F3))
}

equilibrium_jac <- function(x, parms) {
  K_AB <- parms[4]
  K_BC <- parms[5]
  alpha <- parms[6]

  A <- exp(x[1])
  C <- exp(x[2])
  ABC <- exp(x[3])

  df_dy <- rbind(
    c(A, -K_BC * ABC / (alpha * C), K_BC * ABC / (alpha * C) + ABC),
    c(-K_AB * K_BC * ABC / (alpha * A * C) - K_AB * ABC / (alpha * A),
      -K_AB * K_BC * ABC / (alpha * A * C) - K_BC * ABC / (alpha * C),
      K_AB * K_BC * ABC / (alpha * A * C) + K_BC * ABC / (alpha * C) + K_AB * ABC / (alpha * A) + ABC),
    c(-K_AB * ABC / (alpha * A), C, K_AB * ABC / (alpha * A) + ABC)
  )
  return(df_dy)
}

noncoop_f <- function(parms) {
  A_t <- parms[1]
  B_t <- parms[2]
  C_t <- parms[3]
  K_AB <- parms[4]
  K_BC <- parms[5]

  A <- A_t - (A_t + B_t + K_AB - sqrt((A_t + B_t + K_AB)**2 - 4 * A_t * B_t)) / 2
  C <- C_t - (C_t + B_t + K_BC - sqrt((C_t + B_t + K_BC)**2 - 4 * C_t * B_t)) / 2

  phi_AB <- A_t - A
  phi_BC <- C_t - C
  ABC <- ifelse(B_t == 0, 0, phi_AB * phi_BC / B_t)
  return(c(A, C, ABC))
}

################################################################################
## TESTING ROOT-FINDING ##
################################################################################
test_roots <- function(parms) {
  noncoop_sols <- noncoop_f(head(parms, -1))
  init_guess <- log(noncoop_sols)
  equilibrium_roots <- nleqslv(
    x = init_guess,
    fn = equilibrium_f,
    # jac = equilibrium_jac,
    parms = parms
  )
  coop_sols <- exp(equilibrium_roots$x)
  print(noncoop_sols)
  print(coop_sols)
}

test_roots(c(5e-11, 7.5e-2, 1e-5, 5e-8, 1e-9, 1))

test_roots(c(5e-11, 1.2e-2, 1e-5, 5e-8, 1e-9, 1))

test_roots(c(5e-11, 9.18e-9, 1e-5, 5e-8, 1e-9, 1))

test_roots(c(5e-11, 6.24e-15, 1e-5, 5e-8, 1e-9, 1))

test_roots(c(5e-11, 1.12e-6, 1e-5, 1.8e-6, 2.5e-7, 1))

test_roots(c(5e-11, 1.12e-6, 1e-5, 1.8e-6, 2.5e-7, 30))
################################################################################
## NIMBLE FUNCTIONS ##
################################################################################
wrap_solve <- function(A_t, B_t, C_t, K_AB, K_BC, alpha) {
  parms <- c(A_t, B_t, C_t, K_AB, K_BC, alpha)
  init_guess <- noncoop_f(head(parms, -1))
  equilibrium_roots <- nleqslv(
    x = init_guess,
    fn = equilibrium_f,
    jac = equilibrium_jac,
    parms = parms
  )
  return(equilibrium_roots$x[3])
}

nimble_solve <- nimbleRcall(
  prototype = function(A_t = double(0),
                       B_t = double(0),
                       C_t = double(0),
                       K_AB = double(0),
                       K_BC = double(0),
                       alpha = double(0)) {},
  returnType = double(0),
  Rfun = "wrap_solve"
)

################################################################################
## NANOBRET DATA ##
################################################################################
nanobret_csv <- read.csv("data/corrected_nanobret_df.csv")

sorted_min <- sort(unique(nanobret_csv$Minutes))
sorted_min[6:7]

nanobret_96 <- nanobret_csv %>% filter(Minutes %in% sorted_min[6:7])
nanobret_96$mBU_corrected <- nanobret_96$mBU_corrected - min(nanobret_96$mBU_corrected)

ggplot(nanobret_96, aes(x = uM, y = mBU_corrected, color = Construct)) +
  geom_point() +
  scale_x_log10()

nanobret_96 <- nanobret_96 %>% mutate(construct_int = recode(Construct,
  "VHL_WT SMARCA2_L1415S" = 1,
  "VHL_WT SMARCA2_E1420S" = 2,
  "VHL_R69Q SMARCA2_WT" = 3,
  "VHL_Y112F SMARCA2_WT" = 4,
  "VHL_WT SMARCA2_WT" = 5
))

head(nanobret_96)
################################################################################
## NANOBRET NIMBLE MODEL ##
################################################################################
# nimble code
code <- nimbleCode({
  A_t ~ dgamma(.001, .001)
  C_t ~ dgamma(.001, .001)
  relative_constraint ~ dconstraint(A_t < C_t)
  # ratio_constraint ~ dconstraint(0.1 <= (A_t / C_t))
  beta ~ dgamma(.001, .001)
  kappa ~ dunif(0, 1)
  for (i in 1:n_constructs) {
    alpha[i] ~ dgamma(mean = 30, sd = 3)
  }

  for (i in 1:N) {
    B_t[i] <- B_x[i] * kappa
    ABC[i] <- nimble_solve(A_t, B_t[i], C_t, K_AB, K_BC, alpha[construct_int[i]])
    mu[i] <- ABC[i] * beta
    y[i] ~ dnorm(mean = mu[i], sd = 0.05)
  }
})

# list of fixed constants
constants <- list(
  n_constructs = length(unique(nanobret_96$construct_int)),
  B_x = nanobret_96$uM,
  construct_int = nanobret_96$construct_int,
  N = nrow(nanobret_96),
  K_AB = 1.8e-6,
  K_BC = 2.5e-7
)

# list specifying model data
data <- list(
  y = nanobret_96$mBU_corrected,
  relative_constraint = 1
  # ratio_constraint = 1
)

# list specifying initial values
inits <- list(
  A_t = 1,
  C_t = 2,
  beta = 1,
  kappa = 0.75,
  alpha = rep(30, length(unique(nanobret_96$construct_int)))
)

# build the R model object
Rmodel <- nimbleModel(
  code = code,
  constants = constants,
  data = data,
  inits = inits
)
Rmodel$calculate()

################################################################################
### MCMC configuration and building ##
################################################################################
conf <- configureMCMC(Rmodel)

Rmcmc <- buildMCMC(conf)

################################################################################
### compile to C++, and run ###
################################################################################
Cmodel <- compileNimble(Rmodel)
Cmcmc <- compileNimble(Rmcmc, project = Rmodel)

################################################################################
## run multiple MCMC chains ###
################################################################################
samples <- runMCMC(Cmcmc, niter = 10000, nburnin = 1000, samplesAsCodaMCMC = TRUE)
samplesSummary(samples)

samplesPlot(samples)

effectiveSize(samples)

nanobret_96_1 <- nanobret_96 %>% filter(construct_int == 1)
for (B_t in nanobret_96_1$uM) {
  parms <- c(.697, B_t, 2.594, 1.8e-6, 2.5e-7, 27.327)
  noncoop_sols <- noncoop_f(head(parms, -1))
  init_guess <- log(noncoop_sols + 1)
  equilibrium_roots <- nleqslv(
    x = init_guess,
    fn = equilibrium_f,
    # jac = equilibrium_jac,
    parms = parms
  )
  print(exp(equilibrium_roots$x))
  # ABC <- wrap_solve(A_t = 0.697, B_t = B_t, C_t = 2.594, K_AB = 1.8e-6, K_BC = 2.5e-7, alpha = 27.327)
  # pred <- ABC * 7.437
  # print(ABC)
  # print(pred)
}

parms <- c(.697, 5.0, 2.594, 1.8e-6, 2.5e-7, 27.327)
init_guess <- noncoop_f(head(parms, -1))
init_guess
log(init_guess)
equilibrium_roots <- nleqslv(
  x = log(init_guess),
  fn = equilibrium_f,
  # jac = equilibrium_jac,
  parms = parms,
  jacobian = TRUE
)

equilibrium_jac(x = c(2.916e-7, 2.695e-7, .362), parms = parms)

equilibrium_roots$fvec
exp(equilibrium_roots$x)

wrap_solve(A_t = 0.697, B_t = 5, C_t = 2.594, K_AB = 1.8e-6, K_BC = 2.5e-7, alpha = 27.327)

ggplot(nanobret_96_1, aes(x = uM, y = mBU_corrected)) +
  geom_point() +
  scale_x_log10()

# run 3 chains of the crossLevel MCMC
samplesList <- runMCMC(Cmcmc, niter = 1000, nchains = 3)

lapply(samplesList, dim)
