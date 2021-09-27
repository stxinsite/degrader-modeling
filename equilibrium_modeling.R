library(rootSolve)
library(nimble)

equilibriumSys <- function(x, parms){
  A_t = parms[1]
  C_t = parms[2]
  K_AB = parms[3]
  K_BC = parms[4]
  alpha = parms[5]
  B_t = parms[6]

  A = x[1]
  C = x[2]
  ABC = x[3]

  F1 = A + K_BC * ABC / (alpha * C) + ABC - A_t
  F2 = K_AB * K_BC * ABC / (alpha * A * C) + K_BC * ABC / (alpha * C) + K_AB * ABC / (alpha * A) + ABC - B_t
  F3 = C + K_AB * ABC / (alpha * A) + ABC - C_t

  return(c(F1 = F1, F2 = F2, F3 = F3))
}

noncoopEquilibrium <- function(parms){
  A_t = parms[1]
  C_t = parms[2]
  K_AB = parms[3]
  K_BC = parms[4]
  B_t = parms[5]

  A <- A_t - (A_t + B_t + K_AB - sqrt((A_t + B_t + K_AB)**2 - 4*A_t*B_t)) / 2
  C <- C_t - (C_t + B_t + K_BC - sqrt((C_t + B_t + K_BC)**2 - 4*C_t*B_t)) / 2

  phi_AB <- A_t - A
  phi_BC <- C_t - C
  ABC <- ifelse(B_t == 0, 0, phi_AB * phi_BC / B_t)
  return(c(A, C, ABC))
}

parms <- c(5e-11, 1e-5, 1.8e-6, 2.5e-7, 1, 1.12e-6)
init <- noncoopEquilibrium(c(5e-11, 1e-5, 1.8e-6, 2.5e-7, 1.12e-6))
init

equilibriumRoots <- multiroot(equilibriumSys, init, parms = parms)
equilibriumRoots$root

code <- nimbleCode({
  multiroot(equilibriumSys, noncoopEquilibrium(c(5e-11, 1e-5, 1.8e-6, 2.5e-7, 1.12e-6)), c(5e-11, 1e-5, 1.8e-6, 2.5e-7, 1, 1.12e-6))
    for (i in 1:G) {
        a[i] ~ dgamma(1, .001)
        b[i] ~ dgamma(1, .001)
        for (j in 1:N) {
            r[i,j] ~ dbin(p[i,j], n[i,j])
            p[i,j] ~ dbeta(a[i], b[i])
        }
        mu[i] <- a[i] / (a[i] + b[i])
        theta[i] <- 1 / (a[i] + b[i])
    }
})

# list of fixed constants
constants <- list(G = 2,
                  N = 16,
                  n = matrix(c(13, 12, 12, 11, 9, 10, 9,  9, 8, 11, 8, 10, 13,
                      10, 12, 9, 10,  9, 10, 5,  9,  9, 13, 7, 5, 10, 7,  6,
                      10, 10, 10, 7), nrow = 2))

# list specifying model data
data <- list(r = matrix(c(13, 12, 12, 11, 9, 10,  9, 9, 8, 10, 8, 9, 12, 9,
                 11, 8, 9,  8,  9,  4, 8,  7, 11, 4, 4, 5 , 5, 3,  7, 3,
                 7, 0), nrow = 2))

# list specifying initial values
inits <- list(a = c(1, 1),
              b = c(1, 1),
              p = matrix(0.5, nrow = 2, ncol = 16),
              mu    = c(.5, .5),
              theta = c(.5, .5))

# build the R model object
Rmodel <- nimbleModel(code = code,
                      constants = constants,
                      data      = data,
                      inits     = inits)

###########################################
##### MCMC configuration and building #####
###########################################

# generate the default MCMC configuration;
# only wish to monitor the derived quantity "mu"
mcmcConf <- configureMCMC(Rmodel, monitors = "mu")

# check the samplers assigned by default MCMC configuration
mcmcConf$printSamplers()

# double-check our monitors, and thinning interval
mcmcConf$printMonitors()

# build the executable R MCMC function
mcmc <- buildMCMC(mcmcConf)

# let's try another MCMC, as well,
# this time using the crossLevel sampler for top-level nodes

# generate an empty MCMC configuration
# we need a new copy of the model to avoid compilation errors
Rmodel2 <- Rmodel$newModel()
mcmcConf_CL <- configureMCMC(Rmodel2, nodes = NULL, monitors = "mu")

# add two crossLevel samplers
mcmcConf_CL$addSampler(target = c("a[1]", "b[1]"), type = "crossLevel")
mcmcConf_CL$addSampler(target = c("a[2]", "b[2]"), type = "crossLevel")

# let's check the samplers
mcmcConf_CL$printSamplers()

# build this second executable R MCMC function
mcmc_CL <- buildMCMC(mcmcConf_CL)


###################################
##### compile to C++, and run #####
###################################

# compile the two copies of the model
Cmodel <- compileNimble(Rmodel)

Cmodel2 <- compileNimble(Rmodel2)

# compile both MCMC algorithms, in the same
# project as the R model object
# NOTE: at this time, we recommend compiling ALL
# executable MCMC functions together
Cmcmc <- compileNimble(mcmc, project = Rmodel)
Cmcmc_CL <- compileNimble(mcmc_CL, project = Rmodel2)

# run the default MCMC function,
# and example the mean of mu[1]
Cmcmc$run(1000)
cSamplesMatrix <- as.matrix(Cmcmc$mvSamples) # alternative: as.list
mean(cSamplesMatrix[, "mu[1]"])

# run the crossLevel MCMC function,
# and examine the mean of mu[1]
Cmcmc_CL$run(1000)
cSamplesMatrix_CL <- as.matrix(Cmcmc_CL$mvSamples)
mean(cSamplesMatrix_CL[, "mu[1]"])


###################################
#### run multiple MCMC chains #####
###################################

# run 3 chains of the crossLevel MCMC
samplesList <- runMCMC(Cmcmc_CL, niter=1000, nchains=3)

lapply(samplesList, dim)
