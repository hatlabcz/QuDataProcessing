from typing import Tuple, Any, Optional, Union, Dict, List

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import lmfit
from QuDataProcessing.fitter.fitter_base import Fit, FitResult
from QuDataProcessing.helpers.unit_converter import freqUnit, rounder, realImag2magPhase
from QuDataProcessing.fitter.hybrid_freq import HybridFreq


plt.close("all")


fixed_modes = [2]
couplings = [0.2]

s_freq_list = (np.cos(np.linspace(0, np.pi*2, 1001))*0.3+2)*1.4


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

phi_list = np.linspace(0, np.pi, 1001)
plt.figure()
plt.plot(phi_list, bare_mode_freqs[0], "--", label="$f_1$")
plt.plot(phi_list, bare_mode_freqs[1], "--", label="$f_0$")
plt.plot(phi_list, dressed_mode_freqs[0], label="$f_a$")
plt.plot(phi_list, dressed_mode_freqs[1], label="$f_b$")
plt.legend()

# plt.figure()
# plt.plot(phi_list, dressed_mode_freqs[1]-dressed_mode_freqs[0])
# plt.vlines(phi_list[np.argmin(dressed_mode_freqs[1]-dressed_mode_freqs[0])], 0, 5, "grey", "--")
# plt.hlines(couplings[0]*2, phi_list[0], phi_list[-1], "grey", "--")



# dressed_mode_freqs[0][:-240] = [np.nan]* len(dressed_mode_freqs[0][240:])
# dressed_mode_freqs[1][-250:] = [np.nan]* len(dressed_mode_freqs[1][:250])
#
# plt.figure()
# plt.plot(phi_list, bare_mode_freqs[0], "--", label="$f_1$")
# plt.plot(phi_list, bare_mode_freqs[1], "--", label="$f_0$")
# plt.plot(phi_list, dressed_mode_freqs[0], label="$f_a$")
# plt.plot(phi_list, dressed_mode_freqs[1], label="$f_b$")
# plt.legend()



# # fitting
# fit = HybridFreq(dressed_mode_freqs[0], dressed_mode_freqs[3])
# result = fit.run(nan_policy="omit")
# print("guess:", fit.guess(fit.coordinates, fit.data))
# print("fit:", result.params)
# result.plot()
# print()
#
#
