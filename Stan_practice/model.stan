data {
  int<lower=1> K;          // number of mixture components
  int<lower=1> N;          // number of data points
  
  vector[N] real d_obs;
  vector[N] <lower=0> sigma_g_prior;  //
  vector[N] <lower=0> sigma_c_prior;  //

  vector[N] real A_prior;   //
  vector[N] real B_prior;   //
  vector[N] real C_prior;   //

  vector[N] real xg_data;  //
  vector[N] real xc_data;  //
  vector[N] <lower=0> sigma_g_prior;  //
  vector[N] <lower=0> sigma_c_prior;
  vector[N] real A;   //
  vector[N] real B;   //
  vector[N] real C;   //
  vector[N] intersection_x;
  vector[N] intersection_y;


}
parameters {
  simplex[K] theta;          // mixing proportions
  ordered[K] mu;             // locations of mixture components
  vector<lower=0>[K] sigma;  // scales of mixture components

  vector[N] xg;
  vector[N] xc;
  vector[N] d;
}

transformed_parameters{
  vector[N] xg;
  vector[N] xc;
  
  for(n in 1:N):
    d[n]=(((intersection_y[i]-xg[i])/(intersection_x[i]-xc[i]))/sqrt(((intersection_y[i]-xg[i])/(intersection_x[i]-xc[i])))**2)*sqrt((intersection_y[i]-xg[i])**2+(intersection_x[i]-xc[i])**2);

}

model {
  vector[K] log_theta = log(theta);  // cache log calculation
  sigma ~ lognormal(0, 2);
  mu ~ normal(0, 10);
  for (n in 1:N) {
    vector[K] lps = log_theta;
    for (k in 1:K) {
      lps[k] += normal_lpdf(d_obs[n] | mu[k], sqrt((sigma[k]+B_prior[n]**2*sigma_g_prior[n]**2+A_prior[n]**2*sigma_c_prior[n]**2)/(A[n]**2+B[n]**2));
    }
    target += log_sum_exp(lps);
  }

  xg ~ normal(xg_data,sigma_g_prior)
  xc ~ normal(xg_data,sigma_g_prior)

}