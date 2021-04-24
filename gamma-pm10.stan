//
// This Stan program defines a simple model, with a
// vector of values 'y' modeled as normally distributed
// with mean 'mu' and standard deviation 'sigma'.
//
// Learn more about model development with Stan at:
//
//    http://mc-stan.org/users/interfaces/rstan.html
//    https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started
//

data {
    int<lower=0> n;    // number of rows
    int<lower=0> p;    // number of vars
    int a;
    int b;
    matrix[n, p] x;    // prepare the matrix whose first column is all 1s
    int<lower=0> y[n]; // cannot use vector because Y must be integers
}

parameters {
  real alpha;
  vector[p] beta;
}

transformed parameters {
  vector[n] mu;
  mu = alpha + multiply(x, beta);  // matrix multiplication
}

model {
  y ~ poisson_log(mu);
  beta ~ normal(a, b);
}

generated quantities {
  int<lower=0> yhat[n];
  vector[n] log_lik;
  for (i in 1:n) {
    yhat[i] = poisson_log_rng(mu[i]);
    log_lik[i] = poisson_log_lpmf(y[i]|mu[i]);
  }
}
