
# Correcting Predictions for Approximate Bayesian Inference.

Bayesian models quantify uncertainty and facilitate optimal decision-making in downstream applications. For most models, however, practitioners are forced to use approximate inference techniques that lead to sub-optimal decisions due to incorrect posterior predictive distributions. We present a novel approach that corrects for inaccuracies in posterior inference by altering the decision-making process. We train a separate model to make optimal decisions under the approximate posterior, combining interpretable Bayesian modeling with optimization of direct predictive accuracy in a principled fashion. The solution is generally applicable as a plug-in module for predictive decision-making for arbitrary probabilistic programs, irrespective of the posterior inference strategy. We demonstrate the approach empirically in several problems (see below), confirming its potential.


## Publication

The code was used in and is necessary to reproduce results from the paper:

T. Kuśmierczyk, J. Sakaya, A. Klami: **Correcting Predictions for Approximate Bayesian Inference.** [(see arXiv preprint)](https://arxiv.org/pdf/1909.04919.pdf)


## Demonstrations

The main code used for the paper can be found in the following Jupyter notebooks:

 * [normal_normal_mf.ipynb](normal_normal_mf.ipynb)  - Predictive Performance & Posterior Representation: In this notebook, we model last.fm data with Probabilistic Matrix Factorization model (implemented in pyTorch). We compare performance of a traditional approach vs. a decision-maker (neural network) to show that it is capable of correcting errors due to posterior approximation. The inference is carried out using automatic differentiation variational inference (ADVI), and we split the data randomly into equally sized training and test set.

 * [radon.ipynb](radon.ipynb) - Multilevel Models with Poor Approximation: In this notebook, we use the radon data and the multi-level model for modeling it. Next, we train a decision-maker (neural network) to show that it is capable of correcting errors due to unconverged posterior approximation. The model is implemented using the publicly available [Stan code](http://mc-stan.org/users/documentation/case-studies/radon.html), using the Minnesota subset of the data and also otherwise conforming to their details. The inference is carried out using automatic differentiation variational inference, and we split the data randomly into equally sized training and test set.

 * [horseshoe_regression.ipynb](horseshoe_regression.ipynb) - Horseshoe Priors: Sparse Models with Failure of Convergence: In this notebook, we use Bayesian regression with horseshoe priors to model [corn data](https://core.ac.uk/download/pdf/397803.pdf). The model is implemented using the publicly available [Stan code](https://github.com/yao-yl/Evaluating-Variational-Inference/blob/master/R_code/glm_bernoulli_rhs.stan) (additional details can be found in our [paper](https://arxiv.org/pdf/1909.04919.pdf)). For this model the quality of the variational approximation is sensitive to random initialization and the stochastic variation during the optimization, so that occasionally the posterior is reasonable whereas for some runs it converges toa very bad solution. We fix it with help of a decision-maker (neural network) showing that it is capable of correcting errors due to posterior approximation.

 * [linear-reg-bootstrap.ipynb](linear-reg-bootstrap.ipynb) (implemented by Joseph H. Sakaya) - Model Faithfulness: In the notebook, we train a linear model with a decision-maker for highly non-linear data to show that it can learn to directly map the non-linear data, without conforming to the linear model. We demonstrate that a sufficiently complex decision-maker may learn to surpass model predictions, but the effect can be alleviated with regularization.


## Code
The notebooks rely on the following files:

 #### (Stan) Models specification and posterior sampling code:
 * [normal_normal_mf.py](normal_normal_mf.py)
 * [radon.py](radon.py)
 * [horseshoe_regression.py](horseshoe_regression.py)
 
 #### Definition of losses:
 * [losses.py](losses.py)
 
 #### Regularization:
 * [regression_regularization.py](regression_regularization.py)
 
 #### Data loading and preprocessing:
 * [count_data.py](count_data.py)
 * [regression_data.py](regression_data.py)
 
 #### Auxiliary files
 * aux_optimization.py  
 * aux_plt.py  
 * aux.py  


## Prerequisites 
We used Python 3.6.8 with the following libraries:
 * tensorflow 1.14
 * torch  1.1.0
 * scipy 1.2.1
 * numpy  1.16.2
 * pystan 2.19
 * seaborn 
 * matplotlib 
 * pandas 

