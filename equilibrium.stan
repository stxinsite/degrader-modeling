functions {
  vector noncoop_sols(vector parms) {
    vector[3] sols;  // solutions [A], [C], [ABC] to non-cooperative equilibrium system

    real A_t = parms[1];
    real B_t = parms[2];
    real C_t = parms[3];
    real K_AB = parms[4];
    real K_BC = parms[5];
  
    real A = A_t - ((A_t + B_t + K_AB - sqrt(square(A_t + B_t + K_AB) - 4 * A_t * B_t)) / 2);
    real C = C_t - ((C_t + B_t + K_BC - sqrt(square(C_t + B_t + K_BC) - 4 * C_t * B_t)) / 2);
  
    real phi_AB = A_t - A;
    real phi_BC = C_t - C;
    real ABC;
    if (B_t == 0) {
      ABC = 0;  // verify that root-finding transformation handles 0
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
    // y_trans: f(y) = [A], [C], [ABC]
    // initial guesses y should be close to f-inv([A], [C], [ABC])
    vector[3] z;  // the system of mass-action equations
    vector[3] y_trans;  // transform y to constrain [A], [C], [ABC] >= 0
    real A;
    real C;
    real ABC;

    real A_t = theta[1];
    real B_t = theta[2];
    real C_t = theta[3];
    real alpha = theta[4];
    real K_AB = x_r[1];
    real K_BC = x_r[2];
    
    y_trans = square(y);  // f(y) = y^2
    A = y_trans[1];
    C = y_trans[2];
    ABC = y_trans[3];

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
  real<lower=0> kappa = 10;
  real<lower=0> beta = 1;
  real<lower=0> alpha[N_construct] = {25., 30., 25., 25., 30.};
}

parameters {
  real<lower=0> C_t;  // total intracellular E3 concentration
  // real<lower = 0.2 * C_t, upper = 0.5 * C_t> A_t;  // total intracellular target protein concentration
  // real<lower=0> alpha[N_construct];  // alpha_c for each construct c
  // real<lower=0, upper=1> kappa;
  // real<lower=0> beta;
  real<lower=0> sigma;  // error in mBU 
}

transformed parameters {
  real A_t = 0.4 * C_t;  // for now, set A_t deterministic
  // real<lower=0> sigma = 0.5;  // error in mBU 
}

model {
  C_t ~ normal(15, 5);
  // for (j in 1:N_construct) {
  //   alpha[j] ~ normal(30, 5);  // mean = 30, sd = 5
  // }
  // // kappa ~ uniform(0, 1);
  // // beta ~ gamma(.001, .001);
  sigma ~ gamma(.001, .001);
  
  print(A_t);
  print(C_t);

  for (i in 1:N) {
    real B_t;
    vector[3] y_init;
    vector[4] theta;
    vector[3] solutions;
    vector[3] coop_solutions;
    real predmBU;
    
    B_t = B_x[i] * kappa;
    print(B_t);
    theta = [A_t, B_t, C_t, alpha[construct[i]]]';  // arguments for cooperative system root-finding

    y_init = sqrt(noncoop_sols([A_t, B_t, C_t, K_Ds[1], K_Ds[2]]'));  // inverse of f(y) in system()
    print(square(y_init));
    
    coop_solutions = square(algebra_solver_newton(system, y_init, theta, K_Ds, x_i)); 
    
    predmBU = coop_solutions[3] * beta;  // predicted mBU = ABC * beta
    print(predmBU);
    mBU[i] ~ normal(predmBU, sigma);
    // mBU[i] ~ normal(predmBU, sigma);
  }
}
