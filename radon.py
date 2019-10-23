import numpy as np

hierarchical_intercept = """
data {
  int<lower=0> J; 
  int<lower=0> N; 
  int<lower=1,upper=J> county[N];
  vector[N] u;
  vector[N] x;
  vector[N] y;
} 
parameters {
  vector[J] a;
  vector[2] b;
  real mu_a;
  real<lower=0,upper=100> sigma_a;
  real<lower=0,upper=100> sigma_y;
} 
transformed parameters {
  vector[N] y_hat;
  vector[N] m;

  for (i in 1:N) {
    m[i] <- a[county[i]] + u[i] * b[1];
    y_hat[i] <- m[i] + x[i] * b[2];
  }
}
model {
  mu_a ~ normal(0, 1);
  a ~ normal(mu_a, sigma_a);
  b ~ normal(0, 1);
  y ~ normal(y_hat, sigma_y);
}
"""


def create_data(log_radon_train, n_county, county_train, u_train, floor_measure_train):
    hierarchical_intercept_data = {'N': len(log_radon_train),
                          'J': len(n_county),
                          'county': county_train+1, # Stan counts starting at 1
                          'u': u_train,
                          'x': floor_measure_train,
                          'y': log_radon_train}
    return hierarchical_intercept_data


def yield_data_bootstrap(log_radon_train, n_county, county_train, u_train, floor_measure_train):
    log_radon_train, county_train, u_train, floor_measure_train = np.array(log_radon_train), np.array(county_train), np.array(u_train), np.array(floor_measure_train)
    while True:
        ixs = list(range(len(log_radon_train)))
        ixs = np.random.choice(ixs, len(ixs), replace=True)    
        yield create_data(log_radon_train[ixs], n_county, county_train[ixs], u_train[ixs], floor_measure_train[ixs])


def sample_predictive_y0_hmc(hmc_fit, x_test, u_test, county_test, nsamples=1000):
    s = hmc_fit.extract()
    N = s["a"].shape[0]    
    nsamples = min(nsamples, N)
    sample_ixs = np.random.choice(range(N), nsamples, replace=False)
    
    ys = []
    for n in range(nsamples):
        sample_ix = sample_ixs[n]
        sample_a = s["a"][sample_ix][county_test]
        sample_b = s["b"][sample_ix]
        sample_b1 = sample_b[0]
        sample_b2 = sample_b[1]
        sample_yhat = sample_a + u_test*sample_b1 + x_test*sample_b2
        sample_sigma_y = s["sigma_y"][sample_ix]
        sample_y = np.random.normal(sample_yhat, sample_sigma_y)
        ys.append(sample_y)
    return np.array(ys)


def sample_predictive_y0_vi(vi_fit, x_test, u_test, county_test, nsamples=1000, J=85):    
    paramname2ix = dict((paramname, ix) for ix, paramname in enumerate(vi_fit["sampler_param_names"]))
    s = lambda paramname: vi_fit["sampler_params"][paramname2ix[paramname]] # access samples   
    N = len(s("a[1]"))
    nsamples = min(nsamples, N)
    sample_ixs = np.random.choice(range(N), nsamples, replace=False)    
        
    ys = []        
    a_samples = np.array([s("a[%i]" % i) for i in range(1,J+1)])    
    for n in range(nsamples):
        sample_ix = sample_ixs[n]
        sample_a = a_samples[:,sample_ix][county_test]
        sample_b1 = s("b[1]")[sample_ix]
        sample_b2 = s("b[2]")[sample_ix]
        sample_yhat = sample_a + u_test*sample_b1 + x_test*sample_b2
        sample_sigma_y = s("sigma_y")[sample_ix]
        sample_y = np.random.normal(sample_yhat, sample_sigma_y)
        ys.append(sample_y)
    return np.array(ys)


