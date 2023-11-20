import warnings
import numpy as np
import scipy as sp
# import h5py
from scipy.integrate import odeint
from scipy.fft import fft as scifft, fftfreq

import matplotlib.pyplot as plt
from QuDataProcessing.helpers.unit_converter import t2f


# def fft(tList, data, tUnit=None, plot=True, plot_ax=None):
#     N = len(tList)
#     T = tList[1] -tList[0]
#     F_data = scifft(data)
#     F_data = 2.0 / N * np.abs(F_data[0:N // 2])
#     F_freq = fftfreq(N, T)[:N // 2]
#     if plot:
#         if plot_ax is None:
#             fig, plot_ax = plt.subplots()
#         plot_ax.plot(F_freq, F_data)
#         plot_ax.set_yscale("log")
#         if tUnit is not None:
#             plot_ax.set_xlabel(f"Freq {t2f(tUnit)}")
#         plot_ax.grid(True)
#
#     return F_freq, F_data

def fft(tList, data, tUnit=None, plot=True, plot_ax=None, zero_padding=1):
    N = len(tList)
    T = tList[1] - tList[0]

    N_padded = N * zero_padding
    data_padded = np.pad(data, (0, (N_padded - N)), 'constant')
    F_data = np.fft.fft(data_padded)
    F_data = 2.0 / N_padded * np.abs(F_data[:N_padded // 2]) * zero_padding
    F_freq = np.fft.fftfreq(N_padded, T)[:N_padded // 2]

    if plot:
        if plot_ax is None:
            fig, plot_ax = plt.subplots()
        plot_ax.plot(F_freq, F_data)
        plot_ax.set_yscale("log")
        if tUnit is not None:
            plot_ax.set_xlabel(f"Freq {t2f(tUnit)}")
        plot_ax.grid(True)

    return F_freq, F_data


def fft_one_freq(tList, data, freq, zero_padding=1):
    """
    Calculate the DFT coefficient at a single frequency.

    Args:
        tList (array-like): Time vector.
        data (array-like): Input signal data.
        freq (float): Target frequency.
        zero_padding (int, optional): Zero padding factor. Default is 1 (no zero padding).

    Returns:
        complex: DFT coefficient at the target frequency.
    """

    # Pad the signal with zeros
    padded_data = np.pad(data, (0, len(data) * zero_padding), mode='constant')

    # Time vector based on the original time step
    t = np.arange(0, len(padded_data), 1) * (tList[1] - tList[0])
    # Calculate the DFT coefficient at the target frequency
    dft_coefficient = np.sum(padded_data * np.exp(-2j * np.pi * freq * t))
    # normalize and add the negative component
    dft_coefficient = np.abs(dft_coefficient / len(data) * 2)

    return dft_coefficient


