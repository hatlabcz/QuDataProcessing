from typing import Tuple, Any, Optional, Union, Dict, List

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import lmfit
from QuDataProcessing.fitter.fitter_base import Fit, FitResult
from QuDataProcessing.helpers.unit_converter import freqUnit, rounder, realImag2magPhase
from QuDataProcessing.fitter.hybrid_freq import HybridFreq


plt.close("all")


fixed_modes = [3, 4, 6]
couplings = [0.2, 1, 2]

s_freq_list = np.sqrt(np.abs(np.cos(np.linspace(0, np.pi, 1001))))*2


def freq2matrix(f0, f_list, g_list):
    mm =  np.zeros((len(g_list)+1, len(g_list)+1))
    mm[0, 0] = f0
    mm[1:, 0] = g_list
    mm[0, 1:] = g_list
    for i, f in enumerate(f_list):
        mm[i+1, i+1] = f
    return mm


def dressed_modes(m):
    """
    find the eigen values of m, return as a sorted array
    """
    egv = LA.eigvals(m)
    egv_ordered = np.sort(egv)
    return egv_ordered


bare_mode_freqs = np.concatenate((s_freq_list, np.repeat(fixed_modes, len(s_freq_list)))).reshape(len(fixed_modes)+1, -1)
dressed_mode_freqs = [dressed_modes(freq2matrix(f_, fixed_modes, couplings)) for f_ in s_freq_list]
dressed_mode_freqs = np.array(dressed_mode_freqs).T

plt.figure()
for i in range(len(bare_mode_freqs)):
    plt.plot(bare_mode_freqs[i])
    plt.plot(dressed_mode_freqs[i], "--")




# fitting
fit = HybridFreq(dressed_mode_freqs[0], dressed_mode_freqs[3])
result = fit.run(nan_policy="omit")
print("guess:", fit.guess(fit.coordinates, fit.data))
print("fit:", result.params)
result.plot()
print()


