"""Based on the code from 
https://github.com/yao-yl/Evaluating-Variational-Inference/blob/master/R_code/Figure_7_Horseshoe.R 
and https://arxiv.org/pdf/1610.05559.pdf
"""

import numpy as np
import torch

CODE = """

data {
  int<lower=0> n;				      // number of observations
  int<lower=0> d;             // number of predictors
  //int<lower=0,upper=1> y[n];	// outputs
  vector[n] y; //NEW
  matrix[n,d] x;				      // inputs
  real<lower=0> scale_icept;	// prior std for the intercept
  real<lower=0> scale_global;	// scale for the half-t prior for tau
  real<lower=0> slab_scale;
  real<lower=0> slab_df;
}

parameters {
  real beta0; // intercept
  vector[d] z; // auxiliary parameter
  real<lower=0> tau;			// global shrinkage parameter
  vector<lower=0>[d] lambda;	// local shrinkage parameter
  real<lower=0> caux; // auxiliary

  real  logsigma; //NEW
}

transformed parameters {
  
  real<lower=0> c;
  vector[d] beta;				// regression coefficients
  vector[n] f;				// latent values
  vector<lower=0>[d] lambda_tilde;

  real  sigma; //NEW
  sigma = exp(logsigma);  

  c = slab_scale * sqrt(caux);
  lambda_tilde = sqrt( c^2 * square(lambda) ./ (c^2 + tau^2* square(lambda)) );
  beta = z .* lambda_tilde*tau;
  f = beta0 + x*beta;
}

model {
  
  z ~ normal(0,1);
  lambda ~ cauchy(0,1);
  tau ~ cauchy(0, scale_global);
  caux ~ inv_gamma(0.5*slab_df, 0.5*slab_df);
  
  beta0 ~ normal(0,scale_icept);
  //y ~ bernoulli_logit(f);
  y ~ normal(f, sigma); //NEW
}


generated quantities {
  // compute log-likelihoods for loo
  vector[n] loglik;
  for (i in 1:n) {
    //loglik[i] = bernoulli_logit_lpmf(y[i] | f[i]);
    loglik[i] = normal_lpdf(y[i] | f[i], sigma);
  }
}

"""




def create_data(X_TRAIN, Y_TRAIN):
    n, d = X_TRAIN.shape
    scale_icept = 10.0
    tau0 = 1/(d-1) * 2/np.sqrt(n) # τ0= 2(n^1/2(d−1))^−1
    scale_global = tau0
    slab_scale = 5
    slab_df = 4 

    stan_data = {"n": n, "d": d, "y": Y_TRAIN, "x": X_TRAIN, 
                 "scale_icept": scale_icept, "scale_global": scale_global,
                 "slab_scale": slab_scale, "slab_df": slab_df}
    return stan_data


def yield_data_bootstrap(X_TRAIN, Y_TRAIN):
    while True:
        ixs = list(range(len(Y_TRAIN)))
        ixs = np.random.choice(ixs, len(ixs), replace=True)
        yield create_data(X_TRAIN[ixs,:], Y_TRAIN[ixs])


def sample_predictive_y0_vi(vi_fit, x_test, nsamples=1000):        
    paramname2ix = dict((paramname, ix) for ix, paramname in enumerate(vi_fit["sampler_param_names"]))
    s = lambda paramname: vi_fit["sampler_params"][paramname2ix[paramname]] # access samples          
    NS = len(s("beta0"))
    if NS<nsamples: print("[sample_predictive_y0_vi] Warning: can not obtain requested number of samples.")
    nsamples = min(NS, nsamples)
    sample_ixs = np.random.choice(range(NS), nsamples, replace=False)    
        
    ys = []        
    beta_samples = np.array([s("beta[%i]" % i) for i in range(1, x_test.shape[1]+1)]) 
    for n in range(nsamples):
        #print("sampling %i/%i" % (n, nsamples))
        sample_ix = sample_ixs[n]
        
        sample_beta = beta_samples[:,sample_ix]
        sample_beta0 = s("beta0")[sample_ix]
        f = sample_beta0 + x_test.dot(sample_beta)
        
        sample_sigma_y = s("sigma")[sample_ix]
        sample_y = np.random.normal(f, sample_sigma_y)        
        #py = torch.distributions.bernoulli.Bernoulli(logits=torch.tensor(f))
        #sample_y = py.sample().detach().numpy().astype(int)
        ys.append(sample_y)
    return np.array(ys)


def sample_predictive_y0_hmc(hmc_fit, x_test, nsamples=1000):        
    sv = hmc_fit.extract()
    s = lambda paramname: sv[paramname] # access samples          
    NS = len(s("beta0"))
    if NS<nsamples: print("[sample_predictive_y0_hmc] Warning: can not obtain requested number of samples.")
    nsamples = min(NS, nsamples)
    sample_ixs = np.random.choice(range(NS), nsamples, replace=False)    
        
    ys = []        
    beta_samples = s("beta").T
    for n in range(nsamples):
        sample_ix = sample_ixs[n]
        
        sample_beta = beta_samples[:,sample_ix]
        sample_beta0 = s("beta0")[sample_ix]
        f = sample_beta0 + x_test.dot(sample_beta)
        
        sample_sigma_y = s("sigma")[sample_ix]
        sample_y = np.random.normal(f, sample_sigma_y)        
        ys.append(sample_y)
    return np.array(ys)


