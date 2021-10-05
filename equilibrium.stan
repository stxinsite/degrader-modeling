functions {
  vector noncoop_sols(vector parms) {
    vector[3] sols;  // solutions [A], [C], [ABC] to non-cooperative equilibrium system

    real A_t = parms[1];
    real B_t = parms[2];
    real C_t = parms[3];
    real K_AB = parms[4];
    real K_BC = parms[5];
  
    real A = A_t - (A_t + B_t + K_AB - sqrt(square(A_t + B_t + K_AB) - 4 * A_t * B_t)) / 2;
    real C = C_t - (C_t + B_t + K_BC - sqrt(square(C_t + B_t + K_BC) - 4 * C_t * B_t)) / 2;
  
    real phi_AB = A_t - A;
    real phi_BC = C_t - C;
    real ABC;
    if (B_t == 0) {
      ABC = 1e-19;
    } else {
      ABC = phi_AB * phi_BC / B_t;
    }
    
    sols[1] = A;
    sols[2] = C;
    sols[3] = ABC;
    return sols;
  }
  
  vector system(vector y, vector theta, data real[] x_r, int[] x_i) {
    // y: unknown variables to solve for
    // theta: parameters [A]_t, [B]_t, [C]_t, alpha
    // x_r: data K_AB, K_BC
    // sols = exp(y): [A], [C], [ABC]
    vector[3] sols;
    vector[3] z;  // the system of mass-action equations
    real A;
    real C;
    real ABC;

    real A_t = theta[1];
    real B_t = theta[2];
    real C_t = theta[3];
    real alpha = theta[4];
    real K_AB = x_r[1];
    real K_BC = x_r[2];

    sols = exp(y);  // exponential transformation of unknowns
    A = sols[1];
    C = sols[2];
    ABC = sols[3];

    z[1] = A + K_BC * ABC / (alpha * C) + ABC - A_t;
    z[2] = K_AB * K_BC * ABC / (alpha * A * C) + K_BC * ABC / (alpha * C) + K_AB * ABC / (alpha * A) + ABC - B_t;
    z[3] = C + K_AB * ABC / (alpha * A) + ABC - C_t;
    return z;
  }
}

data {
  int<lower=0> N;  // number of data points
  int<lower=1> N_construct;  // number of constructs
  real<lower=0> K_Ds[2];  // [K_AB, K_BC]
  vector[N] B_x;  // extracellular [B]_x
  int construct[N];  // array of integers identifying construct of row i
  vector[N] mBU;  // observed mBU
}

transformed data {
  int x_i[0];  // necessary for algebra_solver() call signature
}

parameters {
  real<lower=0> A_t;
  real<lower=0> C_t;
  real<lower=0> alpha[N_construct];  // alpha_c for each construct c
  real<lower=0, upper=1> kappa;
  real<lower=0> beta;
  // real<lower=0> sigma;
}

model {
  A_t ~ gamma(.001, .001);
  C_t ~ gamma(.001, .001);
  for (j in 1:N_construct) {
    alpha[j] ~ gamma(75, 2.5);  // mean = 30, var = 12
  }
  kappa ~ uniform(0, 1);
  beta ~ gamma(.001, .001);
  
  for (i in 1:N) {
    real B_t = B_x[i] * kappa;
    vector[5] parms = [A_t, B_t, C_t, K_Ds[1], K_Ds[2]]';
    vector[3] y_init = noncoop_sols(parms);  // non-cooperative solutions as initial guesses
    vector[3] y_initLn = log(y_init);  // log-transformation
    vector[4] theta = [A_t, B_t, C_t, alpha[construct[i]]]';  // arguments for algebra_solver()
    vector[3] coop_sols = exp(algebra_solver(system, y_initLn, theta, K_Ds, x_i));  // exponentiate for cooperative solutions
    real predmBU = coop_sols[3] * beta;  // mBU = ABC * beta
    mBU[i] ~ normal(predmBU, 0.1);  // mu = mBU, sigma = 0.1
  }
}

generated quantities {
}
