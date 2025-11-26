import utilBasicRun
import matplotlib.pyplot as plt

import numpy as np
import os

from run_benchmark import KSigma

KSigmaObj = KSigma(
    K_coeff=[0.0005995267, 0.00868861],
    B_coeff=[7.11772e-7, 6.514934e-4, 0.11492713],
    anchor=1600,
)

for iso in [0, 100, 200, 400, 800, 1600, 3200, 4800, 6400]:
    k, sigma = KSigmaObj.GetKSigma(iso)
    # k, sigma = KSigmaObj.GetKSigma(1849)
    print(k, sigma)