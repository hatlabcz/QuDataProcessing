from typing import Tuple, Any, Union
import numpy as np
import matplotlib.pyplot as plt
import lmfit
from QuDataProcessing.fitter.fitter_base import Fit
from QuDataProcessing.helpers.unit_converter import rounder




def dressed_hyb_freq(fb, f0, g):
    """
    function that describes the relation between the dressed (measured) frequencies of two coupled modes.
    assuming only one bare mode (mode 1) is moving. Check notebook.hybridized_modes.nb for details.

    :param fb: dressed frequency of the moving mode (e.g. measured snail mode freq)
    :param f0: bare frequency of the fixed-freq mode
    :param g: coupling strength (the g here is actually the g/2pi in the Hamiltonian)
    :return: dressed frequency of the fixed freq mode (e.g. a transmon mode)
    """
    fa = f0 + g**2/(f0 - fb)
    return fa



class HybridFreqResult():
    def __init__(self, lmfit_result: lmfit.model.ModelResult, fb, fa):
        self.lmfit_result = lmfit_result
        self.params = lmfit_result.params

        self.fb = fb
        self.fa = fa
        self.f0 = self.params["f0"].value
        self.g = self.params["g"].value

    def plot_fphi(self, phi_list=None, plot_ax=None, **figArgs):
        """
        plot the data and fitting results as a function of external bias (or whatever horizontal
        axis you swept, e.g. bias current).
        This plots the data in the way that it was originally taken, and put the fitted curve on top
        of the data, assuming the fitted curve has the same horizontal axis as the data.
        But note that the horizontal axis here is not the independent variable for fitting.
        """

        fa_fit = dressed_hyb_freq(self.fb, self.f0, self.g)
        fa_data = self.fa
        phi_list = np.arange(len(fa_data)) if phi_list is None else phi_list

        if plot_ax is None:
            fig_args_ = dict(figsize=(8, 5))
            fig_args_.update(figArgs)
            fig, ax = plt.subplots(1, 1, **fig_args_)
        else:
            ax = plot_ax
        ax.set_title(f"g = {self.g}")
        ax.plot(phi_list, self.fb, "*", label="fb data")
        ax.plot(phi_list, fa_data, "*", label="fa data")
        ax.plot(phi_list, fa_fit, label="fa fit")
        ax.legend()
        plt.show()

    def plot_fafb(self, plot_ax=None, **figArgs):
        """
        plot one dressed mode frequency as a function of the other dressed mode frequency and the
        fitted curve. This shows how the data was fitted.
        """

        fa_fit = dressed_hyb_freq(self.fb, self.f0, self.g)
        fa_data = self.fa

        if plot_ax is None:
            fig_args_ = dict(figsize=(8, 5))
            fig_args_.update(figArgs)
            fig, ax = plt.subplots(1, 1, **fig_args_)
        else:
            ax = plot_ax
        ax.set_title(f"g = {self.g}")
        ax.plot(self.fb, fa_data, "*", label="data")
        ax.plot(self.fb, fa_fit, label="fit")
        ax.set_xlabel("fb")
        ax.set_ylabel("fa")
        ax.legend()
        plt.show()



    def print(self):
        print(f'g/2pi : {rounder(self.g, 5)}+-{rounder(self.params["g"].stderr, 5)}')
        print(f'f0 : {rounder(self.f0, 5)}+-{rounder(self.params["f0"].stderr, 5)}')



class HybridFreq(Fit):
    def __init__(self, coordinates: Union[Tuple[np.ndarray, ...], np.ndarray],
                 data: np.ndarray):
        """ fit the dressed mode frequency of a fixed-freq mode to the dressed frequency of a moving mode.
        """
        self.coordinates = coordinates
        self.data = data
        self.pre_process()


    @staticmethod
    def model(coordinates, f0,  g) -> np.ndarray:
        freqs = dressed_hyb_freq(coordinates, f0,  g)
        return freqs

    @staticmethod
    def guess(coordinates, data):
        f0 = np.nanmean(data)
        delta_a = np.nanmax(data) - np.nanmin(data)
        delta_ab = np.abs(np.nanmean(coordinates) - f0)
        g = np.sqrt(delta_ab * delta_a)

        f0 = lmfit.Parameter("f0", value=f0, min=np.nanmin(data)*0.5, max=np.nanmax(data)*2) # assume freqs are all positive

        g = lmfit.Parameter("g", value=g, min=0, max=g*20)
        return dict(f0=f0, g=g)



    def run(self, *args: Any, **kwargs: Any) -> HybridFreqResult:
        lmfit_result = self.analyze(self.coordinates, self.data, *args, **kwargs)
        return HybridFreqResult(lmfit_result, fb=self.coordinates, fa=self.data)



if __name__ == "__main__":
    from anti_corssing import hybridize_freq

    # plt.close("all")

    """ ------- create some fake data ---------- """
    biasList = np.linspace(-3, 3, 501)
    f0 = 3 # bare freq of the fixed freq mode
    f1 = - 0.31* biasList ** 2 + 6 # bare freq of the moving mode
    g = 0.02
    freqs = hybridize_freq(f0, f1, g) # dressed frequencies, note this function always returns the higher freq one as the first.
    fb = freqs[0]  # dressed freq of the moving mode
    fa = freqs[1]   # dressed freq of the fixed freq mode
    plt.figure()
    plt.title("example mode frequencies")
    plt.plot(biasList, fb, label="dressed b mode")
    plt.plot(biasList, fa, label="dressed a mode")
    plt.plot(biasList, f1, "--", label="bare b mode")
    plt.plot(biasList, [f0]*len(biasList), "--", label="bare a mode")
    plt.legend()

    """---------- add noise to data ----------------"""
    data = freqs + (np.random.rand(*freqs.shape)- 0.5)*0.0002
    # data[0][:100] = [np.nan] * 100
    # data[1][-100:] = [np.nan] * 100
    plt.figure()
    plt.title("noisy example data")
    plt.plot(biasList, data[0, :])
    plt.plot(biasList, data[1, :])
    plt.plot(biasList, dressed_hyb_freq(data[0, :], f0, g)) # double-check our function


    """ --------- fitting ------------- """
    fit = HybridFreq(data[0], data[1])
    result = fit.run(nan_policy="omit")
    # print("guess params:", fit.guess(fit.coordinates, fit.data))
    # print("fit params:", result.params)

    """
    plot one dressed mode frequency as a function of the other dressed mode frequency and the
    fitted curve. This shows how the data was fitted.
    """
    result.plot_fafb()

    """
    plot the data and fitting results as a function of external bias (or whatever horizontal
    axis you swept, e.g. bias current).
    This plots the data in the way that it was originally taken, and put the fitted curve on top
    of the data, assuming the fitted curve has the same horizontal axis as the data.
    But note that the horizontal axis here is not the independent variable for fitting.
    """
    result.plot_fphi(biasList)

    """
    print result
    """
    result.print()




