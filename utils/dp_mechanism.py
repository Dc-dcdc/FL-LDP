import numpy as np
import math
def cal_sensitivity_up(lr, clip):
    return 2 * lr * clip

def Laplace(epsilon, sensitivity, size):
    noise_scale = sensitivity / epsilon
    return np.random.laplace(0, scale=noise_scale, size=size)

def Gaussian_Simple(epsilon, delta, sensitivity, size):
    noise_scale = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
    return np.random.normal(0, noise_scale, size=size)

# todo
def Gaussian_moment(epsilon, delta, sensitivity, size):
    return