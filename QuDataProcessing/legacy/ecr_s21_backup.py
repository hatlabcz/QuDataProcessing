from typing import Tuple, Any, Optional, Union, Dict, List

import numpy as np
import matplotlib.pyplot as plt
import lmfit
from lmfit.model import ModelResult
from tqdm import tqdm
from QuDataProcessing.fitter.fitter_base import Fit, FitResult
from QuDataProcessing.helpers.unit_converter import freqUnit, rounder, realImag2magPhase
from QuDataProcessing.analyzer.cut_peak import cut_peak

TWOPI = 2 * np.pi
PI = np.pi




def ecr_f21(coordinates, Ql, fn, sn, c1, c2):
    """"
    end coupled resonator
    Kurpiers et al. EPJ Quantum Technology  (2017) 4:8
    DOI 10.1140/epjqt/s40507-017-0059-7"""
    s21 = sn/np.sqrt(1 + 4 * (coordinates/fn-1)**2 * Ql**2) + c1 + c2 * coordinates
    return s21


class ECR_S21Result():
    def __init__(self, lmfit_result: lmfit.model.ModelResult):
        self.lmfit_result = lmfit_result
        self.params = lmfit_result.params
        self.Ql = self.params["Ql"].value
        self.fn = self.params["fn"].value
        self.kappa_2pi = self.fn / self.Ql
        self.freqData = lmfit_result.userkws[lmfit_result.model.independent_vars[0]]

    def plot(self, title='s21  dB', plot_ax=None, **figArgs):
        mag_fit = 20*np.log10(self.lmfit_result.best_fit)
        mag_data = 20*np.log10(self.lmfit_result.data.real)

        fig_args_ = dict(figsize=(6, 5))
        fig_args_.update(figArgs)

        if plot_ax is None:
            fig, ax = plt.subplots(**fig_args_)
        else:
            ax = plot_ax
        ax.set_title(title)
        ax.plot(self.freqData, mag_data, '.')
        ax.plot(self.freqData, mag_fit)
        plt.show()
        
        return ax

    def print(self):

        print(f'fn: {rounder(self.fn, 5)}+-{rounder(self.params["fn"].stderr, 5)}')
        print(f'Ql: {rounder(self.Ql, 5)}+-{rounder(self.params["Ql"].stderr, 5)}')
        print(f'kappa_int/2pi (MHz): {self.fn/self.Ql/1e6}')


class ECR_S21(Fit):
    def __init__(self, coordinates: Union[Tuple[np.ndarray, ...], np.ndarray],
                 data: np.ndarray):
        """ fit cavity reflection function
        :param conjugate: fit to conjugated cavity reflection function (for VNA data)
        """
        self.coordinates = coordinates
        self.data = data
        self.pre_process()

    def model(self, coordinates, Ql, fn, sn, c1, c2) -> np.ndarray:
        """"reflection function of a harmonic oscillator"""
        s21 = ecr_f21(coordinates, Ql, fn, sn, c1, c2)
        return s21

    @staticmethod
    def guess(coordinates, data):
        freq = coordinates

        c2Guess =  (data[-1] - data[0])/(freq[-1]-freq[0])
        snGuess = data[0]
        fnGuess = freq[np.argmax(data)]
        QlGuess = fnGuess / (freq[-1]-freq[0]) * 4
        c1Guess = data[0]

        Ql = lmfit.Parameter("Ql", value=QlGuess, min=QlGuess / 100, max=QlGuess * 100)
        sn = lmfit.Parameter("sn", value=snGuess, min=0)
        fn = lmfit.Parameter("fn", value=fnGuess, min=freq[0], max=freq[-1])
        c1 = lmfit.Parameter("c1", value=c1Guess)
        c2 = lmfit.Parameter("c2", value=c2Guess)

        return dict(Ql=Ql, sn=sn, c1=c1, fn=fn, c2=c2)

    def run(self, *args: Any, **kwargs: Any) -> ECR_S21Result:
        lmfit_result = self.analyze(self.coordinates, self.data, *args, **kwargs)
        return ECR_S21Result(lmfit_result)


class ECR_S21_MultiMode(Fit):
    def __init__(self, coordinates: Union[Tuple[np.ndarray, ...], np.ndarray],
                 data: np.ndarray, nmodes):
        """ fit cavity reflection function
        :param conjugate: fit to conjugated cavity reflection function (for VNA data)
        """
        raise NotImplementedError("This module is not finished, might be incorrect actually.")
        self.coordinates = coordinates
        self.data = data
        self.nmodes = nmodes
        self.pre_process()

        model = lmfit.Model(ecr_f21, prefix='p1_')
        for i in range(1, nmodes):
            model = model + lmfit.Model(ecr_f21, prefix= f"p{i+1}_")
        self.model = model


    def guess(self, coordinates, data):
        freq = coordinates

        guess_dict = {}
        for i in range(self.nmodes):
            c2Guess = (data[-1] - data[0]) / (freq[-1] - freq[0])
            snGuess = data[0]
            fnGuess = freq[np.argmax(data)]
            QlGuess = fnGuess / (freq[-1] - freq[0]) * 4
            c1Guess = data[0]

            Ql = lmfit.Parameter(f"p{i+1}_Ql", value=QlGuess, min=QlGuess / 100, max=QlGuess * 100)
            c2 = lmfit.Parameter(f"p{i+1}_c2", value=c2Guess)
            sn = lmfit.Parameter(f"p{i+1}_sn", value=snGuess, min=0)
            fn = lmfit.Parameter(f"p{i+1}_fn", value=fnGuess, min=freq[0], max=freq[-1])
            c1 = lmfit.Parameter(f"p{i+1}_c1", value=c1Guess)
            guess_dict.update({f"p{i+1}_Ql": Ql, f"p{i+1}_c2": c2,
                               f"p{i+1}_c1": c1, f"p{i+1}_sn": sn,
                               f"p{i+1}_fn": fn})
            new_data, cut_idx_l, cut_idx_r = cut_peak(data, 0.6, plot=False)
            data = new_data
            print(f"{freq[cut_idx_l]}, {freq[cut_idx_r]}")
        print(guess_dict)
        return guess_dict

    def run(self, *args: Any, **kwargs: Any):
        lmfit_result = self.analyze(self.coordinates, self.data, *args, **kwargs)
        return ECR_S21Multi_Result(lmfit_result, nmodes=self.nmodes)


class ECR_S21Multi_Result():
    def __init__(self, lmfit_result: lmfit.model.ModelResult, nmodes):
        self.lmfit_result = lmfit_result
        self.params = lmfit_result.params
        self.nmodes=nmodes
        for i in range(nmodes):
            self.__setattr__(f"p{i+1}_Ql", self.params[f"p{i+1}_Ql"].value)
            self.__setattr__(f"p{i+1}_fn", self.params[f"p{i+1}_fn"].value)
        self.freqData = lmfit_result.userkws[lmfit_result.model.independent_vars[0]]

    def plot(self, **figArgs):
        mag_fit = 20*np.log10(self.lmfit_result.best_fit)
        mag_data = 20*np.log10(self.lmfit_result.data.real)

        fig_args_ = dict(figsize=(6, 5))
        fig_args_.update(figArgs)
        plt.figure(**fig_args_)
        plt.title('s21  dB')
        plt.plot(self.freqData, mag_data, '.')
        plt.plot(self.freqData, mag_fit)
        plt.show()

    def print(self):
        for i in range(self.nmodes):
            print(f'p{i+1}_fn": {rounder(self.__getattribute__(f"p{i+1}_fn"), 5)}+-'
                  f'{rounder(self.params[f"p{i+1}_fn"].stderr, 5)}')
            print(f'p{i+1}_Ql": {rounder(self.__getattribute__(f"p{i+1}_Ql"), 5)}+-'
                  f'{rounder(self.params[f"p{i+1}_Ql"].stderr, 5)}')


def discrete_nl2_fit(freq_data, s21_mag, peak0_region, peak0_n=1, plot=True, window_size=None):
    """
    fit the Qs of the nλ/2 modes in a strip resonator
    :param freq_data: freq in hertz
    :param s21_mag: s21 mag in linear unit
    :param peak0_region: [peak0_start_freq, peak0_stop_freq] region of the first peak
    :param peak0_n: mode index n of the first peak in data
    :param plot: when True, plot results
    :param window_size: window size for peak fitting, in unit of the auto guessed kappa
    :return:
    """
    # -------- fit first peak -----------------
    valid_idx = np.where((freq_data > peak0_region[0]) & (freq_data < peak0_region[1]))[0]
    fit = ECR_S21(freq_data[valid_idx], s21_mag[valid_idx])
    fit_result = fit.run()
    if plot:
        fit_result.plot(title="first peak fitting")
    f0 = fit_result.fn
    kappa0 = fit_result.kappa_2pi

    if window_size is None:
        window_size_ = fit_result.Ql/200
    else:
        window_size_ = window_size

    # # --------- fit all peaks -------------
    fn_list = []
    Qn_list = []
    Qn_std_list = []

    if plot:
        fig, ax = plt.subplots(figsize=(15, 6))
    fn = f0 - f0/peak0_n
    kappan = kappa0
    done = False
    while True:
        for j in range(2):  # two passes
            if j == 0:  # first try to fit with guessed region
                f_start, f_stop = fn + peak0_region[0]- f0 + f0/peak0_n, fn + peak0_region[1]- f0 + f0/peak0_n
            else:  # fit based on the first fitting results
                f_start, f_stop = fn - kappan * window_size_, fn + kappan * window_size_

            if (f_start + f_stop)/2 > freq_data[-1]:
                done = True
                break

            valid_idx = np.where((freq_data > f_start) & (freq_data < f_stop))[0]
            fit = ECR_S21(freq_data[valid_idx], s21_mag[valid_idx])
            fit_result = fit.run()
            fn = fit_result.fn
            Qn = fit_result.Ql
            kappan = fn / Qn  # kappa/2pi

            if window_size is None:
                window_size_ = fit_result.Ql/200
            # if np.abs(fn - f0 * (i + 1)) >= 10 * kappa0:
            #     print(fit.guess(fit.coordinates, fit.data))
            #     raise RuntimeError

        if done:
            break

        fn_list.append(fn)
        Qn_list.append(Qn)
        Qn_std_list.append(fit_result.lmfit_result.params["Ql"].stderr)
        if plot:
            fit_result.plot(plot_ax=ax)

    print(Qn_std_list)

    if plot:
        fig.tight_layout()
        plt.figure(figsize=(6, 5))
        plt.errorbar(np.array(fn_list) / 1e9, Qn_list, yerr=Qn_std_list)
        plt.xlabel("freq (GHz)")
        plt.ylabel("Q")

    return np.array(fn_list), np.array(Qn_list), np.array(Qn_std_list)



if __name__ == '__main__':
    pass