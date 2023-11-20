from typing import Tuple, Any, Optional, Union, Dict, List

import json
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import json
from scipy.optimize import minimize_scalar

from QuDataProcessing.base import Analysis, AnalysisResult
from QuDataProcessing.fitter.generic_functions import Lorentzian
from QuDataProcessing.analyzer.cut_peak import cut_peak
from QuDataProcessing.analyzer.extract_mode_freq import extract_mode_freq

plt.close("all")


# --------------- Extract Qubit Trace ---------------------------------
data = json.load(open(r"L:\Data\SNAIL_Pump_Limitation\QubitMSMT\pulseSpec_sweepBias\2023-01-03\\Q_0.825mA_freq_3520_3520_bias_-0.00254_0.00108processed"))
x_data = np.array(data["biasList"])
freq_list = np.array(data["freqList"])
q_data = np.array(data["glist"])[:,:,0]


n_modes = 1
mode_freqs = extract_mode_freq(freq_list, q_data, n_modes)

plt.figure()
plt.pcolormesh(x_data, freq_list, q_data.T, shading="auto")
for i in range(n_modes):
    plt.plot(x_data,mode_freqs[i], color="r")



# --------------- Read SNAIL Trace ---------------------------------

import h5py
from scipy.interpolate import  interp1d
def loadModeFreq(fileDir, fileName, plot=True):
    f = h5py.File(fileDir + fileName, 'r')
    currList = f['currList'][()]
    modeFreqList = f['modeFreqList'][()]
    if plot:
        plt.figure()
        plt.title("expData")
        plt.plot(currList, modeFreqList)
    return currList, modeFreqList


filepath = r'L:\Data\SNAIL_Pump_Limitation\SNAIL_FluxSweep\20230102\\'
filename = 'SNAIL_-45-30dBm_Avg40_-2.4to1mA_3.7to6.5GHz'
curr_, modeFreq_ = loadModeFreq(filepath, filename + '_modeFreq', plot=False)
curr_ *= 1e3
modeFreq_ /=1e6
plt.figure()
plt.plot(curr_[:], modeFreq_[:])

f_s_itp = interp1d(curr_[:], modeFreq_[:], fill_value=np.nan)
f_s = np.zeros_like(x_data) + np.nan
for i, x in enumerate(x_data):
    try:
        f_s[i] = f_s_itp(x)
    except ValueError:
        pass

plt.plot(x_data, f_s, ".")



# ---------------  Fitting  ---------------------------------
from QuDataProcessing.fitter.hybrid_freq import HybridFreq
fit = HybridFreq(f_s, mode_freqs[0])
result = fit.run(nan_policy="omit", g=20)
print(result.params)
result.plot(x_data)
result.print()

plt.figure()
plt.plot(f_s, mode_freqs[0], "*")
plt.plot(f_s[:-1], result.lmfit_result.best_fit)




