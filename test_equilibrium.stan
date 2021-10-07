functions {
  vector noncoop_sols(vector parms) {
    vector[3] sols;  

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
      ABC = 0;
    } else {
      ABC = phi_AB * phi_BC / B_t;
    }
    
    sols[1] = A;
    sols[2] = C;
    sols[3] = ABC;
    return sols;
  }
  
  vector system(vector y, vector theta, data real[] x_r, int[] x_i) {
    vector[3] z;  
    vector[3] y_trans;
    real A;
    real C;
    real ABC;

    real A_t = theta[1];
    real B_t = theta[2];
    real C_t = theta[3];
    real alpha = theta[4];
    real K_AB = x_r[1];
    real K_BC = x_r[2];
    
    y_trans = square(y);
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
  int<lower=0> N;  
  int<lower=1> N_construct;  
  real<lower=0> K_Ds[2];  
  vector[N] B_x;  
  int construct[N];  
}

transformed data {
  int x_i[0]; 
}

parameters {}

transformed parameters {
  real<lower=0> C_t;
  real<lower = 0.2 * C_t, upper = 0.5 * C_t> A_t;
  real<lower=0> alpha[N_construct] = {20., 25., 30., 35., 40.};
}

model {
  for (i in 1:N) {
    real B_t;
    vector[5] parms;
    vector[3] noncoop_solutions;
    vector[3] y_init;
    vector[4] theta;
    vector[3] solutions;
    vector[3] coop_solutions;

    
    B_t = B_x[i];
    parms = [A_t, B_t, C_t, K_Ds[1], K_Ds[2]]';
    theta = [A_t, B_t, C_t, alpha[construct[i]]]';  
    print(theta);

    noncoop_solutions = noncoop_sols(parms);  
    print(noncoop_solutions);
    y_init = sqrt(noncoop_solutions);  
    solutions = algebra_solver(system, y_init, theta, K_Ds, x_i);
    print(solutions);
    coop_solutions = square(solutions);
    print(coop_solutions);
  }
}

generated quantities {
  
}
