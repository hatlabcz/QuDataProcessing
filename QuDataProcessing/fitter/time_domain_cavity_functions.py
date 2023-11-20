from typing import Tuple, Any, Optional, Union, Dict, List

import numpy as np
from numpy.fft import ifft
import matplotlib.pyplot as plt
import lmfit
from lmfit.model import ModelResult
from QuDataProcessing.fitter.fitter_base import Fit, FitResult
from QuDataProcessing.helpers.unit_converter import freqUnit, rounder, realImag2magPhase

TWOPI = 2 * np.pi
PI = np.pi


@np.vectorize
def cav_response_ref(t, kint, kext, delta_f, t0, pw):
    """
    time domain cavity response function for single port reflection msmt.
    The formula is calculated using the inverse fourier transform of the frequency domain response.

    :param t: time (s)
    :param kint: kappa internal. Note: this is kappa, not kappa/2pi
    :param kext: kappa external. Note: this is kappa, not kappa/2pi
    :param delta_f: drive detuning (Hz)
    :param t0: drive pulse start time (s)
    :param pw: drive pulse width (s)
    :return:
    """
    dd = delta_f * TWOPI
    denom = kext + kint - 2j * dd

    if t < t0:
        return 0j
    elif t < t0 + pw:
        nume = (-1 + 2 * np.exp((t0 - t) / 2 * denom)) * kext + kint - 2j * dd
        return nume / denom
    else:
        nume = np.exp((t0 - t) / 2 * denom) - np.exp((t0 + pw - t) / 2 * denom)
        nume *= 2*kext
        return nume / denom


def cav_response_ref_decay(t, k, delta_f):
    """
    decay part of the time domain cavity response for single port reflection msmt.
    The formula is calculated using the inverse fourier transform of the frequency domain response.

    :param t: time (s)
    :param k: kappa total. Note: this is kappa, not kappa/2pi
    :param delta_f: drive detuning (Hz)
    :return:
    """
    dd = delta_f * TWOPI
    denom = k - 2j * dd
    return np.exp(- t / 2 * denom)



class CavTraceRefResult():
    def __init__(self, lmfit_result: lmfit.model.ModelResult):
        self.lmfit_result = lmfit_result
        self.params = lmfit_result.params

        self.kext = self.params["kext"].value
        self.kint = self.params["kint"].value
        self.timeData = lmfit_result.userkws[lmfit_result.model.independent_vars[0]]

    def plot(self, plot_ax=None, **figArgs):
        I_fit = self.lmfit_result.best_fit.real
        Q_fit = self.lmfit_result.best_fit.imag
        I_data = self.lmfit_result.data.real
        Q_data = self.lmfit_result.data.imag

        if plot_ax is None:
            fig_args_ = dict(figsize=(8, 5))
            fig_args_.update(figArgs)
            fig, ax = plt.subplots(1, 1, **fig_args_)
        else:
            ax = plot_ax

        ax.plot(self.timeData, I_data,  label="I data")
        ax.plot(self.timeData, Q_data,  label="Q data")
        ax.plot(self.timeData, I_fit, label="I fit")
        ax.plot(self.timeData, Q_fit, label="Q fit")
        ax.legend()
        plt.show()

    def print(self):
        print(f'kext/2pi: {rounder(self.kext/TWOPI, 5)}+-{rounder(self.params["kext"].stderr/TWOPI, 5)}')
        print(f'kint/2pi: {rounder(self.kint/TWOPI, 5)}+-{rounder(self.params["kint"].stderr/TWOPI, 5)}')

        print(f'delta_f (MHz): {rounder(self.params["delta_f"].value/1e6, 5)}+-{rounder(self.params["delta_f"].stderr/1e6, 5)}')
        print(f't0 (us): {rounder(self.params["t0"].value*1e6, 3)}+-{rounder(self.params["t0"].stderr*1e6, 3)}')
        print(f'pw (us): {rounder(self.params["pw"].value*1e6, 3)}+-{rounder(self.params["pw"].stderr*1e6, 3)}')
        print(f'A: {rounder(self.params["A"].value, 3)}+-{rounder(self.params["A"].stderr, 3)}')
        print(f'phi: {rounder(self.params["phi"].value, 3)}+-{rounder(self.params["phi"].stderr, 3)}')

class CavTraceRef(Fit):
    def __init__(self, coordinates: Union[Tuple[np.ndarray, ...], np.ndarray],
                 data: np.ndarray, conjugate: bool = False):
        """ fit cavity reflection function
        :param conjugate: fit to conjugated cavity reflection function (for VNA data)
        """
        self.coordinates = coordinates
        self.data = data
        self.conjugate = conjugate
        self.pre_process()

    def model(self, coordinates, kint, kext, delta_f, t0, pw, A, phi) -> np.ndarray:
        """"reflection function of a harmonic oscillator"""
        trace = cav_response_ref(coordinates, kint, kext, delta_f, t0, pw)
        if self.conjugate:
            trace = trace.conjugate()
        trace *= A * np.exp(1j * phi)
        return trace

    @staticmethod
    def guess(coordinates, data):
        t_trace = coordinates
        I_trace = np.real(data)
        Q_trace = np.imag(data)
        abs_trace = np.abs(data)
        npts = len(I_trace)

        # todo: better guess functions for these parameters..
        kint_guess = 5e3 * TWOPI
        kext_guess = 3e6 * TWOPI
        delta_f_guess = 1e6

        # todo: this t0 guess might not be very robust..
        zero_abs = np.average(abs_trace[-10:])
        zero_std = np.std(abs_trace[-10:])
        idxes = np.where(abs_trace > zero_abs + zero_std * 5)[0]
        for idx in idxes:
            if np.average(abs_trace[idx+1: idx+5])> zero_abs + zero_std * 5:
                t0_idx=idx
                break

        t0_guess = t_trace[t0_idx]
        t0 = lmfit.Parameter("t0", value=t0_guess, vary=False)

        A_guess = np.max(abs_trace)
        phi_guess = np.angle(I_trace[npts//2] + 1j* Q_trace[npts//2])

        kint = lmfit.Parameter("kint", value=kint_guess, min=10 * TWOPI, max=1e8 * TWOPI)
        kext = lmfit.Parameter("kext", value=kext_guess, min=1e2 * TWOPI, max=1e8 * TWOPI)
        delta_f = lmfit.Parameter("delta_f", value=delta_f_guess, min=-20e6, max=20e6)
        A = lmfit.Parameter("A", value=A_guess, min=-A_guess*3, max=A_guess*3)
        phi = lmfit.Parameter("phi", value=phi_guess, min=0, max=TWOPI)

        return dict(kint=kint, kext=kext, delta_f=delta_f, t0=t0, A=A, phi=phi)



    def run(self, *args: Any, **kwargs: Any) -> CavTraceRefResult:
        if "pw" not in kwargs:
            raise ValueError("the current fitting function requires feeding in pulse width (pw, in s)."
                             " e.g. pass 'pw=config['res_pulse_config']['length']*1e-6' ")

        kwargs["pw"] = lmfit.Parameter("pw", value=kwargs["pw"], vary=False)
        lmfit_result = self.analyze(self.coordinates, self.data, *args, **kwargs)
        return CavTraceRefResult(lmfit_result)

class CavTraceRefDecayResult():
    def __init__(self, lmfit_result: lmfit.model.ModelResult):
        self.lmfit_result = lmfit_result
        self.params = lmfit_result.params
        self.k = self.params["k"].value
        self.timeData = lmfit_result.userkws[lmfit_result.model.independent_vars[0]]

    def plot(self, plot_ax=None, **figArgs):
        I_fit = self.lmfit_result.best_fit.real
        Q_fit = self.lmfit_result.best_fit.imag
        I_data = self.lmfit_result.data.real
        Q_data = self.lmfit_result.data.imag

        if plot_ax is None:
            fig_args_ = dict(figsize=(8, 5))
            fig_args_.update(figArgs)
            fig, ax = plt.subplots(1, 1, **fig_args_)
        else:
            ax = plot_ax

        ax.plot(self.timeData, I_data, label="I data")
        ax.plot(self.timeData, Q_data, label="Q data")
        ax.plot(self.timeData, I_fit, label="I fit")
        ax.plot(self.timeData, Q_fit, label="Q fit")
        ax.legend()
        plt.show()

    def print(self):
        print(f'kappa/2pi: {rounder(self.k / TWOPI, 5)}+-{rounder(self.params["k"].stderr / TWOPI, 5)}')
        print(f'delta_f (MHz): {rounder(self.params["delta_f"].value / 1e6, 5)}+-{rounder(self.params["delta_f"].stderr / 1e6, 5)}')
        print(f'A: {rounder(self.params["A"].value, 3)}+-{rounder(self.params["A"].stderr, 3)}')
        print(f'phi: {rounder(self.params["phi"].value, 3)}+-{rounder(self.params["phi"].stderr, 3)}')


class CavTraceRefDecay(Fit):
    def __init__(self, coordinates: Union[Tuple[np.ndarray, ...], np.ndarray],
                 data: np.ndarray, conjugate: bool = False):
        """ fit cavity reflection function
        :param conjugate: fit to conjugated cavity reflection function (for VNA data)
        """
        self.coordinates = coordinates
        self.data = data
        self.conjugate = conjugate
        self.pre_process()

    def model(self, coordinates, k, delta_f, A, phi) -> np.ndarray:
        """"reflection function of a harmonic oscillator"""
        trace = cav_response_ref_decay(coordinates, k, delta_f)
        if self.conjugate:
            trace = trace.conjugate()
        trace *= A * np.exp(1j * phi)
        return trace

    def guess(self, coordinates, data):
        t_trace = coordinates
        abs_trace = np.abs(data)
        phase_trace = np.unwrap(np.angle(data))
        npts = len(t_trace)

        # todo: better guess functions for these parameters..
        k_guess = 3e6 * TWOPI
        delta_f_guess = (phase_trace[npts//2] - phase_trace[0])/(t_trace[npts//2] - t_trace[0])/2/np.pi
        if self.conjugate:
            delta_f_guess *= -1

        A_guess = np.max(abs_trace)
        phi_guess = phase_trace[0]

        k = lmfit.Parameter("k", value=k_guess, min=10 * TWOPI, max=1e8 * TWOPI)
        delta_f = lmfit.Parameter("delta_f", value=delta_f_guess, min=delta_f_guess-20e6, max=delta_f_guess+20e6)
        A = lmfit.Parameter("A", value=A_guess, min=A_guess * 0.5, max=A_guess * 2)
        phi = lmfit.Parameter("phi", value=phi_guess, min=phi_guess-2, max=phi_guess+2)

        return dict(k=k, delta_f=delta_f, A=A, phi=phi)


    def run(self, *args: Any, **kwargs: Any) -> CavTraceRefDecayResult:
        lmfit_result = self.analyze(self.coordinates, self.data, *args, **kwargs)
        return CavTraceRefDecayResult(lmfit_result)



if __name__ == '__main__':
    plt.close("all")

    # ------------ test fit data------------------------
    # import json
    # g_data = json.load(open("D:\\Temp\\Q3_11.5mA_g_trace"))
    # e_data = json.load(open("D:\\Temp\\Q3_11.5mA_e_trace"))
    # t_trace = np.array(g_data["time_trace"])
    # IQ_trace_g = g_data["I_trace"] + 1j * np.array(g_data["Q_trace"])
    # IQ_trace_e = e_data["I_trace"] + 1j * np.array(e_data["Q_trace"])
    # plt.figure(1, figsize=(10,10))
    # plt.plot(t_trace, np.real(IQ_trace_g), ".", color="C0")
    # plt.plot(t_trace, np.imag(IQ_trace_g), ".", color="C1")
    # plt.plot(t_trace, np.abs(IQ_trace_g), ".", color="C2")
    # # plt.plot(t_trace, np.real(IQ_trace_e))
    # plt.plot(t_trace, np.imag(IQ_trace_e))

    # --------- test cavity response function --------------------
    kint = 3e4 * TWOPI
    kext = 2.6e6 * TWOPI
    delta_f = 0.4e6
    t0 = 2.6e-7
    pw = 2e-6
    A = -24
    phi = np.pi/3 * 0.6
    t_list = np.linspace(0,3.22e-6,501)
    # response = cav_response_ref(t_list, kint, kext, delta_f, t0, pw, A, phi)
    response = cav_response_ref_decay(t_list, kint+ kext, delta_f, A, phi+np.pi)
    plt.figure(1)
    plt.plot(t_list*1e6, np.real(response))
    plt.plot(t_list*1e6, np.imag(response))
    plt.plot(t_list*1e6, np.abs(response))



    # -------------------------- fit
    # fit = CavTraceRef(t_trace*1e-6, IQ_trace_e)
    # # result = fit.run(Qint=2e5, Qext=2e3, f0=6.5e9, delta=0.4e6, t0=2.6e-7, pw=2e-6, A=-24, phi = np.pi/3 * 0.5)
    # result = fit.run(pw=2e-6)
    # result.plot()
    # result.print()



