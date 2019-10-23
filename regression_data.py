import scipy.io
import numpy as np
import pandas as pd


def load_corn(src='data/corn.mat', x_name="mp5spec", shuffle=False):
    # Load data:
    mat = scipy.io.loadmat(src)
    X = mat[x_name][0][0][7]
    Y = mat["propvals"][0][0][7]

    if shuffle:
        np.random.seed(123)
        order = list(range(len(Y)))
        np.random.shuffle(order)
        X, Y = X[order,:], Y[order]

    return X, Y


def load_radon():
    srrs2 = pd.read_csv('data/srrs2.dat')
    srrs2.columns = srrs2.columns.map(str.strip)
    srrs_mn = srrs2.assign(fips=srrs2.stfips*1000 + srrs2.cntyfips)[srrs2.state=='MN']

    cty = pd.read_csv('data/cty.dat')
    cty_mn = cty[cty.st=='MN'].copy()
    cty_mn[ 'fips'] = 1000*cty_mn.stfips + cty_mn.ctfips

    srrs_mn = srrs_mn.merge(cty_mn[['fips', 'Uppm']], on='fips')
    srrs_mn = srrs_mn.drop_duplicates(subset='idnum')
    u = np.log(srrs_mn.Uppm)

    n = len(srrs_mn)
    srrs_mn.county = srrs_mn.county.str.strip()
    mn_counties = srrs_mn.county.unique()
    counties = len(mn_counties)

    county_lookup = dict(zip(mn_counties, range(len(mn_counties))))
    county = srrs_mn['county_code'] = srrs_mn.county.replace(county_lookup).values
    radon = srrs_mn.activity
    srrs_mn['log_radon'] = log_radon = np.log(radon + 0.1).values
    floor_measure = srrs_mn.floor.values
    n_county = srrs_mn.groupby('county')['idnum'].count()
    return log_radon, n_county, county, u, floor_measure  
