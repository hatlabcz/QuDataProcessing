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



def extract_mode_freq(freq_list, spec_data, n_modes=2, fit_tol = 0.15, n_peak_bw = 30):
    """
    extract mode frequencies from a flux sweep (or other parameter sweep) of qubit pulse spec.
    For each flux bias, the pulse spec data is fitted to a Lorentzian and find the qubit mode freq.
    Multiple modes can also be extracted by setting n_modes > 1 (e.g. for anti-crossing data).


    :param freq_list: frequency list for the pulse spec
    :param spec_data: 2D pulse spec data, the first axis should be the flux bias (or other sweep parameters)
    :param n_modes: number of modes to find in each pulse spec.
    :param fit_tol: threshold for the success of the Lorentzian fitting (i.e. for a success fitting, the stderr/value of
        the peak height must be lower than this value)
    :param n_peak_bw: for multi-mode fitting, the minimum peak distance in the unit of peak_bw
    :return:
    """
    results = np.zeros((n_modes, len(spec_data))) + np.nan
    for i, data in enumerate(tqdm(spec_data)):
        new_data = data
        for n_ in range(n_modes):
            fit_ = Lorentzian(freq_list, new_data)
            fit_result_ = fit_.run(nan_policy="omit")
            A_ = fit_result_.params["A"].value
            k_ = fit_result_.params["k"].value
            x0_ = fit_result_.params["x0"].value
            # fit_result_.plot()

            # check if fitting is successful, if True, add fitted data to result list
            fit_success = fit_result_.success and (fit_result_.params["A"].stderr/np.abs(A_) < fit_tol)
            if fit_success:
                results[n_, i] = x0_
                # remove current fitted peak and get ready for the next fitting
                peak_start_idx = np.argmin(np.abs(freq_list-x0_+np.sqrt(n_peak_bw/k_)))
                peak_end_idx = np.argmin(np.abs(freq_list-x0_-np.sqrt(n_peak_bw/k_)))
                peak_region = np.clip([peak_start_idx, peak_end_idx], 0, len(freq_list))
                temp_ = np.arange(0, len(new_data))
                new_data = np.where((temp_ > peak_region[0]) & (temp_ < peak_region[1]), np.nan, new_data)
            else:
                break

    return results


#todo: function that splits the frequnecy traces

# results = np.zeros((n_modes, len(x_data))) + np.nan
# trace0 = np.linspace(0, -90, 91)
# trace1 = np.linspace(0, -90, 91) + 50
#
# results[0, 0: 50] = trace0[0:50]
# results[0, 60:] = trace1[60:]
# results[1, 30:61] = trace1[30:61]
#
# data = results
#
# plt.close("all")
#
# correct_data = np.zeros_like(data) + np.nan
# last_data = data[:,0]
# correct_data[:, 0] = last_data
# for i in range(1, len(x_data)):
#     new_data = data[:, i]
#     for j, d in enumerate(new_data):
#         print(i, j, "d:",d, "last_data:", last_data)
#         if np.isfinite(d):
#             diff = np.abs(d - last_data)
#             new_data_idx = np.nanargmin(diff)
#             correct_data[new_data_idx, i] = d
#
#             print("!!!!!!!", d, correct_data[:, i])
#         else:
#             break
#             # print("?????????", d)
#             # finite_new_data_idx, = np.where(np.isfinite(new_data))
#             # print(finite_new_data_idx)
#             # if len(finite_new_data_idx) != 0:
#             #     correct_data[j, i] = new_data[finite_new_data_idx[0]]
#             #     new_data_idx = finite_new_data_idx[0]
#             # else:
#             #     correct_data[j, i] = np.nan
#             #     break
#         # new_data = np.delete(new_data, new_data_idx)
#         # if len(np.where(np.isfinite(new_data))[0]) == 1:
#         #     break
#         # print(i, j,"new new_data", new_data)
#
#     last_data = correct_data[:, i]
# plt.figure()
# for i in range(n_modes):
#     plt.plot(results[i])
#     plt.plot(correct_data[i], "*")
        

if __name__ == "__main__":
    plt.close("all")

    data = json.load(open(
        r"L:\Data\SNAIL_Pump_Limitation\QubitMSMT\pulseSpec_sweepBias\2023-01-03\\Q_0.825mA_freq_3520_3520-2processed"))

    x_data = np.array(data["biasList"])
    freq_list = np.array(data["freqList"])
    q_data = np.array(data["glist"])[:, :, 0]

    n_modes = 2
    mode_freqs = extract_mode_freq(freq_list, q_data,  n_modes)
    plt.figure()
    plt.pcolormesh(x_data, freq_list, q_data.T, shading="auto")
    for i in range(n_modes):
        plt.plot(x_data, mode_freqs[i], color="r")

    # split freq traces
    f_traces = np.zeros_like(mode_freqs) + np.nan
    f_traces[0, :40] = mode_freqs[0, :40]
    f_traces[1, 40:] = mode_freqs[0, 40:]
    plt.figure()
    plt.pcolormesh(x_data, freq_list, q_data.T, shading="auto")
    for i in range(n_modes):
        plt.plot(x_data,f_traces[i])


