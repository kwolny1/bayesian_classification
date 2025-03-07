import numpy as np
import pandas as pd

def scheme1_dataset(a): 
    n = 1000
    prob = 0.5
    mean_0 = 0
    mean_1 = a
    std_dev = 1

    n_ones = np.sum(np.random.binomial(1, prob, size=n))

    X1_0 = np.random.normal(mean_0, std_dev, n-n_ones)
    X2_0 = np.random.normal(mean_0, std_dev, n-n_ones)
    y_0 = np.zeros(n-n_ones)

    X1_1 = np.random.normal(mean_1, std_dev, n_ones)
    X2_1 = np.random.normal(mean_1, std_dev, n_ones)
    y_1 = np.ones(n_ones)

    df_0_X = pd.DataFrame({'X_1': X1_0, 'X_2': X2_0})
    df_1_X = pd.DataFrame({'X_1': X1_1, 'X_2': X2_1})

    scheme1_X = pd.concat([df_0_X, df_1_X])
    scheme1_y = np.concatenate((y_0, y_1))
    return scheme1_X, scheme1_y

def scheme2_dataset(a, rho): 
    n = 1000
    p = 2
    prob = 0.5

    means = [[0, 0], [a, a]]
    cov_0 = [[1, rho], [rho, 1]]
    cov_1 = [[1, -rho], [-rho, 1]]
    n_ones = np.sum(np.random.binomial(1, prob, size=n))

    ns = [n-n_ones, n_ones]
    X0 = np.random.multivariate_normal(means[0], cov_0, ns[0]).T
    X1 = np.random.multivariate_normal(means[1], cov_1, ns[1]).T
    y3 = [np.zeros(n-n_ones), np.ones(n_ones)]

    df_0 = pd.DataFrame({'X_1': X0[0], 'X_2': X0[1]})
    df_1 = pd.DataFrame({'X_1': X1[0], 'X_2': X1[1]})
    scheme2_X = pd.concat([df_0, df_1])
    scheme2_y = np.concatenate((y3[0], y3[1]))
    return scheme2_X, scheme2_y