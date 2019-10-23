
import time
import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.functional import softplus

import aux_optimization
from aux import flatten_first_two_dims


env = torch if torch.get_default_dtype()==torch.float32 else torch.cuda
    
    
def totorch(array):
    return torch.tensor(array, dtype=env.float32)



def jacobian_softplus(x):
       return 1.0/(1.0 + torch.exp(-x))
    
    
def model_log_prob(x, w, z, mask=None, sgd_scale=1.0):
    if mask is None: mask = torch.ones_like(x).type(env.ByteTensor)
    
    xhat = z.matmul(w)
    likelihood = Normal(xhat, 1.) 
    prior = Normal(0, 10)
    assert likelihood.loc.shape[1: ] == x.shape
    
    return torch.masked_select(likelihood.log_prob(x), mask).sum()*sgd_scale \
            + prior.log_prob(w).sum() + prior.log_prob(z).sum()*sgd_scale


def sample_predictive_y0(qw, qz, nsamples_theta, nsamples_y):  
    """ Returns a tensor with samples     
        (nsamples_y samples of y for each theta x 
         nsamples_theta samples of latent variables)."""
    w = qw.rsample(torch.Size([nsamples_theta]))
    z = qz.rsample(torch.Size([nsamples_theta]))    
    
    xhat = z.matmul(w)
    likelihood = Normal(xhat, 1.)
    y_samples = likelihood.rsample(torch.Size([nsamples_y]))
    return y_samples


def sample_predictive_y(qw, qz, nsamples_theta, nsamples_y):  
    """ Returns a tensor with samples (nsamples_y x nsamples_theta).
        Flattents the first two dimensions 
        (samples of y for different thetas) from sample_predictive_y0.
    """
    return flatten_first_two_dims(sample_predictive_y0(qw, qz, nsamples_theta, nsamples_y))


def vi_inference(Y, TRAIN_MASK, K, MINIBATCH, NSAMPLES, SEED, NITER, LR):
    N, D = Y.shape
    torch.manual_seed(SEED)

    qz_loc = torch.randn([N, K], requires_grad=True)
    qz_scale = torch.randn([N, K], requires_grad=True)
    qw_loc = torch.randn([K, D], requires_grad=True)
    qw_scale = torch.randn([K, D], requires_grad=True)

    optimizer = torch.optim.Adam([qw_loc, qw_scale, qz_loc, qz_scale], lr=LR)
    x = totorch(Y)
    training_mask = torch.tensor(TRAIN_MASK).type(env.ByteTensor)

    start = time.time()    
    for i in range(NITER):
        
            rows, epoch_no, sgd_scale = aux_optimization.yield_minibatch_rows(i, N, MINIBATCH)

            #######################################################        
            # preparation: selecting minibatch rows

            qz_loc0 = qz_loc[rows, :]
            qz_scale0 = qz_scale[rows, :]    
            qw = Normal(qw_loc, softplus(qw_scale))
            qz = Normal(qz_loc0, softplus(qz_scale0))

            x0 = x[rows,:]
            training_mask0 = training_mask[rows, :]

            #######################################################
            # optimization

            w = qw.rsample(torch.Size([NSAMPLES]))
            z = qz.rsample(torch.Size([NSAMPLES]))
            elbo = model_log_prob(x0, w, z, training_mask0, sgd_scale).sum() \
                            -qw.log_prob(w).sum() -qz.log_prob(z).sum()*sgd_scale 
            elbo = elbo/NSAMPLES

            optimizer.zero_grad()            
            objective = -elbo 
            objective.backward(retain_graph=False)
            optimizer.step()    
        
            #######################################################          
            if i%1000==0 or i<10:
              print("[%.2fs] %i. iteration, %i. epoch" % (time.time()-start, i, epoch_no))               

    qw = Normal(qw_loc, softplus(qw_scale))
    qz = Normal(qz_loc, softplus(qz_scale))
    return qw, qz

 
