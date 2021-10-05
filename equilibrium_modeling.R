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
    c(
      -K_AB * K_BC * ABC / (alpha * A * C) - K_AB * ABC / (alpha * A),
      -K_AB * K_BC * ABC / (alpha * A * C) - K_BC * ABC / (alpha * C),
      K_AB * K_BC * ABC / (alpha * A * C) + K_BC * ABC / (alpha * C) + K_AB * ABC / (alpha * A) + ABC
    ),
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
  noncoop_sols[noncoop_sols == 0] <- 1e-19 # replace 0 with small epsilon
  init_guess <- log(noncoop_sols)
  equilibrium_roots <- nleqslv(
    x = init_guess,
    fn = equilibrium_f,
    jac = equilibrium_jac,
    parms = parms
  )
  coop_sols <- exp(equilibrium_roots$x)
  print(noncoop_sols)
  print(coop_sols)
  print(equilibrium_roots$fvec)
}

# UNIT TEST
test_roots(c(5e-11, 0, 1e-5, 5e-8, 1e-9, 1))
test_roots(c(5e-11, 7.5e-2, 1e-5, 5e-8, 1e-9, 1))
test_roots(c(5e-11, 1.2e-2, 1e-5, 5e-8, 1e-9, 1))
test_roots(c(5e-11, 9.18e-9, 1e-5, 5e-8, 1e-9, 1))
test_roots(c(5e-11, 6.24e-15, 1e-5, 5e-8, 1e-9, 1))
test_roots(c(5e-11, 1.12e-6, 1e-5, 1.8e-6, 2.5e-7, 1))
test_roots(c(5e-11, 1.12e-6, 1e-5, 1.8e-6, 2.5e-7, 30))
# as alpha gets larger, ABC increases and reduces A more than C
test_roots(c(5e-11, 1.12e-6, 1e-5, 1.8e-6, 2.5e-7, 50))
################################################################################
## NIMBLE FUNCTIONS ##
################################################################################
wrap_solve <- function(parms) {
  noncoop_sols <- noncoop_f(head(parms, -1))
  noncoop_sols[noncoop_sols == 0] <- 1e-19 # replace 0 with small epsilon
  init_guess <- log(noncoop_sols)
  equilibrium_roots <- nleqslv(
    x = init_guess,
    fn = equilibrium_f,
    jac = equilibrium_jac,
    parms = parms
  )
  return(exp(equilibrium_roots$x[3]))
}

nimble_solve <- nimbleRcall(
  prototype = function(parms = double(1)) {},
  returnType = double(0),
  Rfun = "wrap_solve"
)

testModelCode <- nimbleCode({
    ABC[1] <- nimble_solve(x[1:6])
})

testModel <- nimbleModel(testModelCode, check = FALSE, calculate = FALSE)

ctestModel <- compileNimble(testModel)

## Let's see if it works:
ctestModel$x <- c(5e-11, 1.12e-6, 1e-5, 1.8e-6, 2.5e-7, 50)

## Answer directly from nearPD:
wrap_solve(ctestModel$x)
## Answer via the nimble model:
ctestModel$calculate('ABC[1]')
ctestModel$ABC  #This result should match the answer calculated directly from nearPD.
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

summary(nanobret_96)
################################################################################
## NANOBRET NIMBLE MODEL ##
################################################################################
# nimble code
code <- nimbleCode({
  A_t ~ dgamma(.001, .001)
  C_t ~ dgamma(.001, .001)
  constraint ~ dconstraint(0.1 * C_t <= A_t & A_t <= C_t)
  beta ~ dgamma(.001, .001)
  kappa ~ dunif(0, 1)
  for (i in 1:N_constructs) {
    alpha[i] ~ dgamma(mean = 30, sd = 5)
    sigma[i] ~ dgamma(.001, .001)
  }

  for (i in 1:N) {
    B_t[i] <- B_x[i] * kappa
    ABC[i] <- nimble_solve(c(A_t, B_t[i], C_t, K_AB, K_BC, alpha[construct_int[i]]))
    mu[i] <- ABC[i] * beta
    y[i] ~ dnorm(mean = mu[i], sd = sigma[construct_int[i]])
  }
})

# list of fixed constants
constants <- list(
  N_constructs = length(unique(nanobret_96$construct_int)),
  B_x = nanobret_96$uM,
  construct_int = nanobret_96$construct_int,
  N = nrow(nanobret_96),
  K_AB = 1.8,
  K_BC = .25
)

# list specifying model data
data <- list(
  y = nanobret_96$mBU_corrected,
  constraint = 1
)

# list specifying initial values
# VERIFY UNIFORMITY OF MEASUREMENT UNITS
inits <- list(
  A_t = 1,
  C_t = 2,
  beta = 1,
  kappa = 0.75,
  alpha = rep(30, length(unique(nanobret_96$construct_int))),
  sigma = rep(1, length(unique(nanobret_96$construct_int)))
)

inits_f <- function() {
  A_t <- runif(n = 1, min = 0.5, max = 10)
  list(
    A_t = A_t,
    C_t = 2 * A_t,
    beta = runif(n = 1, min = 1, max = 10),
    kappa = runif(n = 1, min = 0.25, max = 1),
    alpha = rnorm(n = 5, mean = 30, sd = 4),
    sigma = rgamma(n = 5, shape = 1, scale = 1)
  )
}

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
samples <- runMCMC(Cmcmc, niter = 20000, nburnin = 1000, thin = 2, samplesAsCodaMCMC = TRUE)
samples_n3 <- runMCMC(Cmcmc, nchains = 3, niter = 10000, nburnin = 1000, thin = 2, inits = inits_f, samplesAsCodaMCMC = TRUE)

samplesSummary(samples)
lapply(samples_n3, samplesSummary)

effectiveSize(samples)
lapply(samples_n3, effectiveSize)

save(samples, file = "samples.RData")

samplesPlot(samples)

################################################################################
## Plot predicted ABC ##
################################################################################
nanobret_96_1 <- nanobret_96 %>% filter(construct_int == 1)
unique(nanobret_96_1$Construct)

samplesSum <- samplesSummary(samples)
mBU_pred <- numeric(length = nrow(nanobret_96_1))
for (i in 1:nrow(nanobret_96_1)) {
  parms <- c(.051, nanobret_96_1$uM[i],
             .1611, 1.8, .25, 26.9)
  ABC <- wrap_solve(parms = parms)
  mBU_pred[i] <- ABC * samplesSum['beta','Mean']
}

nanobret_96_1$mBU_pred <- mBU_pred

df <- data.frame(uM = rep(nanobret_96_1$uM, 2),
                 mBU = c(nanobret_96_1$mBU_corrected, nanobret_96_1$mBU_pred),
                 type = c(rep("observed", nrow(nanobret_96_1)), rep("predicted", nrow(nanobret_96_1)))
               )
ggplot(df, aes(x = uM, y = mBU, color = type)) +
  geom_point() +
  geom_line() +
  scale_x_log10()
