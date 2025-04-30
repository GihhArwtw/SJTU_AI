import numpy as np
from scipy.integrate import quad

import math

"""
Optionally you could use moments accountant to implement the epsilon calculation.
"""

def E_1(x, q, sigma, sensitivity):
    coef = np.exp(-1 * (x**2) / (2*sigma**2)) / (np.sqrt(2*np.pi) * sigma)
    pow = 1 / (1 - q + q * np.exp((2*sensitivity*x - 1) / (2*sigma**2)))
    return coef * pow

def E_2(x, q, sigma, sensitivity):
    coef = (q * np.exp(-1 * ((x - sensitivity)**2) / (2*sigma**2)) + (1 - q)*np.exp(-1 * (x**2) / (2*sigma**2))) / (np.sqrt(2*np.pi) * sigma)
    pow = (1 - q + q * np.exp((2*sensitivity*x - 1) / (2*sigma**2)))
    return coef * pow

def calculate_epsilon(q, sigma, steps, delta, sensitivity):
    if q == 0:
        alpha = 0
    elif q == 1:
        alpha = steps / (2 * sigma**2)
    else:
        I1 = quad(E_1, -np.inf, np.inf, args=(q, sigma, sensitivity))
        I2 = quad(E_2, -np.inf, np.inf, args=(q, sigma, sensitivity))
        alpha = steps * np.log(max(I1[0], I2[0]))
    return (alpha - math.log(delta))

def get_epsilon(epoch, delta, sigma, sensitivity, batch_size, training_nums):
    """
    Compute epsilon with basic composition from given epoch, delta, sigma, sensitivity, batch_size and the number of training set.
    """
    q = batch_size / training_nums
    steps = int(math.ceil(epoch * training_nums / batch_size))
    return calculate_epsilon(q, sigma, steps, delta, sensitivity)
