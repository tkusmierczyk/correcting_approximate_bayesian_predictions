
from aux import tonumpy
import numpy as np

def get_regularization_constant(ys, l=0.):
    return np.ones(ys.shape[1:])*l


def get_regularization_std(ys, l=0.):
    sigma = tonumpy(ys).std(0) * l
    lam = 1. / sigma
    return lam    


def get_regularization_qdiff(ys, l=0.):
    Quantiles = np.array([np.percentile(tonumpy(ys), int(q*100), axis=0) for q in [0.5-abs(l), 0.5+abs(l)]])
    sigma = abs(Quantiles[0,:]-Quantiles[1,:])
    lam = 1. / sigma
    return lam


def get_regularization1(ys, method, l):
    if method.startswith("const"): return get_regularization_constant(ys, l)
    if method.startswith("std"): return get_regularization_std(ys, l)
    if method.startswith("qdiff"): return get_regularization_qdiff(ys, l)
    raise Exception("Unknown regularization method name: %s" % method)


def _bootstrap_infer_decisions(sm, stan_data_generator, 
                              sample_predictive_y0_train, 
                              sample_predictive_y0_test, 
                              optimal_h_bayes_estimator_np, 
                              iter=10e4, seed=123):
    stan_data = next(stan_data_generator)
    vi_fit = sm.vb(data=stan_data, iter=iter, output_samples=1e3,
                   tol_rel_obj=0.001, eta=0.1, seed=seed, verbose=True)
    
    ys1_train = sample_predictive_y0_train(vi_fit)
    ys1_test = sample_predictive_y0_test(vi_fit)

    hstar1_train = optimal_h_bayes_estimator_np(ys1_train)
    hstar1_test  = optimal_h_bayes_estimator_np(ys1_test)
    return hstar1_train, hstar1_test


def get_bootstrap_decisions(sm, stan_data_generator, 
                            sample_predictive_y0_train,
                            sample_predictive_y0_test, 
                            optimal_h_bayes_estimator_np,
                            num_repetitions=5, iter=10e4, seed=123):
    hstar1_train, hstar1_test = [], []
    for r in range(num_repetitions):
        print("[get_bootstrap_decisions] %i/%i" % (r, num_repetitions))
        h1_train, h1_test = _bootstrap_infer_decisions(sm, stan_data_generator, 
                                 sample_predictive_y0_train,
                                 sample_predictive_y0_test, 
                                 optimal_h_bayes_estimator_np, iter=iter, seed=seed+r) 
        hstar1_train.append(h1_train)
        hstar1_test.append(h1_test)
    hstar1_train = np.array([h1 for h1 in hstar1_train])
    hstar1_test = np.array([h1 for h1 in hstar1_test])
        
    return hstar1_train, hstar1_test
