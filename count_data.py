
import numpy as np
import pandas as pd


def generate_data(I=1000, D=100, K=10, a=0.05, b=0.01, c=1.0, seed=123):
    np.random.seed(seed)
    theta = np.random.gamma(a, 1./(a*c), (I,K))
    beta = np.random.gamma(b, 1./b, (K,D))
    y = np.random.poisson(theta.dot(beta))
    mask = np.ones((I,D))
    return y, mask

  
def lastfm_data(N=1000, D=100, url="data/lastfm_data.csv", log=True):    
    """LastFm views."""
    df = pd.read_csv(url, header=None)
    x_ = df.values #users in rows, artists in colums
    x_ = x_[:N, :D]
    N, D = x_.shape

    if log: x_ = np.log(1+x_)
    mask_ = np.ones( (N,D) ) #non-missing values
    return x_, mask_  


def random_mask(x, testing_prob=0.5, seed=123): 
    np.random.seed(seed)
    N, D = x.shape
    testing_mask = np.random.choice([0, 1], (N, D), (1.0-testing_prob, testing_prob))
    training_mask = np.ones((N, D))-testing_mask
    return training_mask, testing_mask


def test_train_split(Y, MASK, testing_prob = 0.5):
    # Prepares masks to work with pytorch
    TRAIN_MASK, _ = random_mask(Y, testing_prob)
    TRAIN_MASK = TRAIN_MASK.astype(np.uint8)
    TRAIN_MASK[Y.shape[0]-1, Y.shape[1]-1] = 1 # always include bottom-right corner (necessary for sparse implementation)
    TEST_MASK = 1-TRAIN_MASK

    TRAIN_MASK = TRAIN_MASK*MASK
    TEST_MASK = TEST_MASK*MASK
    Y_TRAIN = Y*TRAIN_MASK
    Y_TEST = Y*TEST_MASK

    assert Y_TEST.sum()+Y_TRAIN.sum() - (Y*MASK).sum() < 0.0001
    return Y_TRAIN, Y_TEST, TRAIN_MASK, TEST_MASK


