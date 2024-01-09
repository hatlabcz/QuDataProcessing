from typing import Tuple, Any, Optional, Union, Dict, List
from scipy.optimize import minimize, newton, least_squares
from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt
import lmfit
from lmfit.model import ModelResult, Model, Parameter
from QuDataProcessing.fitter.fitter_base import Fit, FitResult
from QuDataProcessing.helpers.unit_converter import freqUnit, rounder, realImag2magPhase

TWOPI = 2 * np.pi
PI = np.pi

"""
resonator freq domain msmt based on method in this papar
https://arxiv.org/abs/1410.3365
"""


def core_func(f, f0, Ql, Qc):
    """
    core function of for all three different configurations
    :param f: probe freq
    :param f0: mode freq
    :param Ql: total Q
    :param Qc: coupling Q
    :return:
    """
    x = f / f0 - 1
    nume = Ql / Qc
    denom = 1 + 2j * Ql * x
    return nume / denom


def instrument_factor(f, amp, phase_off, e_delay):
    """
    pre factor on the msmt result due to msmt setup
    :param f: probe freq
    :param amp: attenuation, gain in the msmt chain
    :param phase_off: phase offset
    :param e_delay: electrical delay
    :return:
    """
    return amp * np.exp(1j * (phase_off - TWOPI * f * e_delay))


def hanger_func(f, f0, Ql, Qc_m, amp, phase_off, e_delay, phi):
    pre_ = instrument_factor(f, amp, phase_off, e_delay)
    core_ = core_func(f, f0, Ql, Qc_m)
    s21 = pre_ * (1 - core_ * np.exp(1j * phi))
    return s21


def refl_func(f, f0, Ql, Qc, amp, phase_off, e_delay):
    pre_ = instrument_factor(f, amp, phase_off, e_delay)
    core_ = core_func(f, f0, Ql, Qc)
    s11 = pre_ * (1 - 2 * core_)
    return s11


def trans_func(f, f0, Ql, amp, phase_off, e_delay):
    pre_ = instrument_factor(f, amp, phase_off, e_delay)
    core_ = core_func(f, f0, Ql, 1) #Qc absorbed into amp
    s21 = pre_ * 2 * core_
    return s21


def approximate_null_space(A):
    """
    Compute an approximate null space of a matrix A using SVD.
    """
    U, S, Vt = linalg.svd(A)
    null_idx = np.argmin(np.abs(S))
    aprox_space = Vt[null_idx, :].T
    return aprox_space


def fit_circle(x, y):
    """
    fit complex data (x, y) to a circle on the complex plane
    :param x:
    :param y:
    :return:
    """
    z = x ** 2 + y ** 2
    v = np.array([z, x, y, np.ones_like(x)])
    M = np.matmul(v, v.T)
    B = np.zeros((4, 4))
    B[0, 3] = B[3, 0] = -2

    """ 
    this is mathematically correct, and looks efficient, but doesn't work as well as the 
    Newton method, because of machine precision issue... The M matrix can have a huge condition number... 
    
    # M - eta_ * B is a quadratic function (since B only have two non-zero elements),
    # we just need to find the coefficients for this quadratic function
    c2 = 4 * (M[1, 2] * M[2, 1] - M[1, 1] * M[2, 2])
    c0 = linalg.det(M)
    c1 = linalg.det(M - B) - c2 - c0

    # eta_ is minimum positive root for the quadratic function
    c_dis = np.sqrt(c1 ** 2 - 4 * c2 * c0)
    v1, v2 = (-c1 + c_dis) / 2 / c2, (-c1 - c_dis) / 2 / c2
    print(v1, v2)
    eta_ = min([v_ for v_ in [v1, v2] if v_ > 0], default=None)
    """

    def solve_eta(eta):
        return linalg.det(M - eta * B)

    eta_ = newton(solve_eta, 0)

    a, b, c, d = approximate_null_space(M - eta_ * B)
    x0 = - b / 2 / a
    y0 = - c / 2 / a
    r0 = np.sqrt(b ** 2 + c ** 2 - 4 * a * d) / 2 / abs(a)

    return x0, y0, r0


def guess_e_delay(freq, data, debug=False):
    """
    guess electrical delay
    :param freq: frequency array
    :param data: complex data array
    :return:
    """
    phase = np.unwrap(np.angle(data))
    # tau = -(phase[-1] - phase[0]) / (freq[-1] - freq[0]) / TWOPI # only works fine for hanger
    npts = len(freq)
    k1, b1 = np.polyfit(freq[:npts // 10], phase[:npts // 10], 1)
    k2, b2 = np.polyfit(freq[-npts // 10:], phase[-npts // 10:], 1)
    tau = -(k1 + k2) / 2 / TWOPI

    if debug:
        plt.figure()
        plt.plot(freq, phase)
        plt.plot(freq, k1 * freq + b1)
        plt.plot(freq, k2 * freq + b2)
        print("!!!!!!!!!!!!!!!!!! guess e_delay", tau)

    return tau


def fit_e_delay(freq, data, debug=False):
    """
    find the e_delay that makes the curve most closely to a circle
    :param freq: frequency array
    :param data: complex data array
    :return:
    """

    def circ_residual(tau):
        rot_data = np.exp(1j * TWOPI * freq * tau) * data
        x, y = rot_data.real, rot_data.imag
        x0, y0, r0 = fit_circle(x, y)
        return np.sum(abs(r0 ** 2 - (x - x0) ** 2 - (y - y0) ** 2))

    tau0 = guess_e_delay(freq, data, debug)
    tau_fit = least_squares(circ_residual, tau0, method="lm").x[0]

    return tau_fit


def fit_phase(freq, phase, debug=False):
    def phse_func(f, theta0, Ql, f0):
        theta = theta0 - 2 * np.arctan(2 * Ql * (f / f0 - 1))
        return theta

    model = Model(phse_func)
    theta0_gue = (phase[0] + phase[-1]) / 2
    f0_gue = freq[np.argmin(np.abs(phase - theta0_gue))]
    Ql_gue = np.mean(freq) / (np.max(freq) - np.min(freq)) * 5
    params = {}
    params["theta0"] = Parameter("theta0", theta0_gue)
    params["f0"] = Parameter("f0", f0_gue, min=np.min(freq), max=np.max(freq))
    params["Ql"] = Parameter("Ql", Ql_gue, min=Ql_gue / 50, max=Ql_gue * 50)
    if debug:
        print(params)
    fit_result = model.fit(phase, f=freq, **params)

    return fit_result


class ResonatorFit(Fit):
    def __init__(self, coordinates: Union[Tuple[np.ndarray, ...], np.ndarray],
                 data: np.ndarray):
        self.coordinates = coordinates
        self.data = data
        self.pre_process()

    def _fit_circ_and_phase(self, debug=True):
        freq = self.coordinates
        data = self.data

        # fit for e_delay
        self.e_delay = fit_e_delay(freq, data, debug)
        # circle data after fixing e_delay
        data_cir = data * np.exp(1j * self.e_delay * TWOPI * freq)
        # fitted circle
        self.xc, self.yc, self.rc = fit_circle(data_cir.real, data_cir.imag)
        # translate circle data to origin
        data_tr = data_cir - (self.xc + 1j * self.yc)

        # phase fit
        data_tr_phase = np.unwrap(np.angle(data_tr))
        phase_fit_res = fit_phase(freq, data_tr_phase, debug)
        self.theta0 = phase_fit_res.params["theta0"].value
        self.Ql = phase_fit_res.params["Ql"].value
        self.f0 = phase_fit_res.params["f0"].value

        if debug:
            fig, ax = plt.subplots(1, 2, figsize=(12,5))
            ax[0].set_title("circle_fit")
            ax[0].plot(data.real, data.imag, ".")
            ax[0].plot(data_cir.real, data_cir.imag, ".")
            theta_ = np.linspace(0, TWOPI, 1001)
            ax[0].plot(self.xc + self.rc * np.cos(theta_), self.yc + self.rc * np.sin(theta_))
            ax[0].plot(data_tr.real, data_tr.imag, ".")
            ax[0].set_aspect(1)
            ax[1].set_title("phase fit")
            ax[1].plot(freq, data_tr_phase, ".")
            ax[1].plot(freq, phase_fit_res.best_fit)


class ResonatorResult():
    def __init__(self, freq, data, params):
        self.freq = freq
        self.data = data
        self.params = params
        self.model = params.pop("model")
        self.f0 = params["f0"]
        self.Ql = params["Ql"]

    def plot(self, plot_axes=None, **figArgs):
        fitted_data = self.model(self.freq, **self.params)
        real_fit = fitted_data.real
        imag_fit = fitted_data.imag
        mag_fit, phase_fit = realImag2magPhase(real_fit, imag_fit)
        mag_data, phase_data = realImag2magPhase(self.data.real, self.data.imag)

        fig_args_ = dict(figsize=(12, 5))
        fig_args_.update(figArgs)

        if plot_axes is None:
            fig, ax = plt.subplots(1, 2, **fig_args_)
        else:
            ax = plot_axes
        ax[0].set_title('mag (dB pwr)')
        ax[0].plot(self.freq, mag_data, '.')
        ax[0].plot(self.freq, mag_fit)
        ax[1].set_title('phase')
        ax[1].plot(self.freq, phase_data, '.')
        ax[1].plot(self.freq, phase_fit)
        plt.show()

    def print(self):
        print(f'f (Hz): {self.f0:.6e}')
        print(f'Qtot: {self.Ql:.3e}')
        if "Qc_m" in self.params:
            print(f'|Qc|: {self.params["Qc_m"]:.3e}')
        for k in ["Qc", "Qi"]:
            try:
                print(f'{k}: {self.__getattribute__(k):.3e}')
            except AttributeError:
                pass
        print('T1 (us):', f"{(self.Ql / self.f0 / 2 / np.pi * 1e6):.3e}")
        try:
            print('MaxT1 (us):', f"{self.Qi / self.f0 / 2 / np.pi * 1e6:.3e}")
        except AttributeError:
            pass
        print('kappa_tot/2Pi: ', f"{self.f0 / self.Ql / 1e6 :.4e}", 'MHz')
        try:
            print('kappa_c/2Pi: ', f"{self.f0 / self.Qc / 1e6 :.4e}", 'MHz')
        except AttributeError:
            pass


class HangerFit(ResonatorFit):
    def __init__(self, coordinates: Union[Tuple[np.ndarray, ...], np.ndarray],
                 data: np.ndarray):
        super().__init__(coordinates, data)

    @staticmethod
    def model(coordinates, f0, Ql, Qc_m, amp, phase_off, e_delay, phi) -> np.ndarray:
        s21 = hanger_func(coordinates, f0, Ql, Qc_m, amp, phase_off, e_delay, phi)
        return s21

    def extract_params(self):
        beta = self.theta0 + PI
        x_ = self.xc + self.rc * np.cos(beta)
        y_ = self.yc + self.rc * np.sin(beta)
        phase_off = np.angle(x_ + 1j*y_)
        amp = np.sqrt(x_**2 + y_**2)
        phi = beta - phase_off

        Qc_m = self.Ql * amp/ (2 * self.rc)
        Qc = Qc_m/np.cos(phi)
        Qi = 1/(1/self.Ql - 1/Qc)


        params = {"f0": self.f0, "Ql": self.Ql, "Qc_m": Qc_m, "amp": amp, "phase_off": phase_off,
                  "e_delay": self.e_delay, "phi": phi, "Qc": Qc, "Qi": Qi, "model":self.model}

        return params

    def run(self, debug=True):
        self._fit_circ_and_phase(debug)
        params = self.extract_params()
        return HangerResult(self.coordinates, self.data, params)


class HangerResult(ResonatorResult):
    def __init__(self, freq, data, params):
        self.Qc = params.pop("Qc")
        self.Qi = params.pop("Qi")
        super().__init__(freq, data, params)



class ReflFit(ResonatorFit):
    def __init__(self, coordinates: Union[Tuple[np.ndarray, ...], np.ndarray],
                 data: np.ndarray):
        super().__init__(coordinates, data)

    @staticmethod
    def model(coordinates, f0, Ql, Qc, amp, phase_off, e_delay) -> np.ndarray:
        s21 = refl_func(coordinates, f0, Ql, Qc, amp, phase_off, e_delay)
        return s21

    def extract_params(self):
        beta = self.theta0 + PI
        x_ = self.xc + self.rc * np.cos(beta)
        y_ = self.yc + self.rc * np.sin(beta)
        phase_off = np.angle(x_ + 1j*y_)
        amp = np.sqrt(x_**2 + y_**2)
        Qc = self.Ql * amp/ (self.rc)
        Qi = 1/(1/self.Ql - 1/Qc)

        params = {"f0": self.f0, "Ql": self.Ql, "Qc": Qc, "amp": amp, "phase_off": phase_off,
                  "e_delay": self.e_delay, "Qi": Qi, "model":self.model}

        return params
    
    def re_fit(self, guess_params):
        model = Model(self.model)
        fit_params = model.make_params()
        for k, v in fit_params.items():
            v.value = guess_params[k]
            v.min = v.value - 0.1 * np.abs(v.value)
            v.max = v.value + 0.1 * np.abs(v.value)

        fit_result = model.fit(self.data, coordinates=self.coordinates, **fit_params)
        return fit_result

    
    def run(self, debug=True, refit=False):
        self._fit_circ_and_phase(debug)
        params = self.extract_params()

        if refit:
            fit_result = self.re_fit(params)
            for k, v in fit_result.params.items():
                params[k] = v.value
            params["Qi"] = 1 / (1 / params["Ql"] - 1 / params["Qc"])

        return ReflResult(self.coordinates, self.data, params)


class ReflResult(ResonatorResult):
    def __init__(self, freq, data, params):
        self.Qc = params["Qc"]
        self.Qi = params.pop("Qi")
        super().__init__(freq, data, params)



class TransFit(ResonatorFit):
    def __init__(self, coordinates: Union[Tuple[np.ndarray, ...], np.ndarray],
                 data: np.ndarray):
        super().__init__(coordinates, data)

    @staticmethod
    def model(coordinates, f0, Ql, amp, phase_off, e_delay) -> np.ndarray:
        s21 = trans_func(coordinates, f0, Ql, amp, phase_off, e_delay)
        return s21

    def extract_params(self):
        params = {"f0": self.f0, "Ql": self.Ql, "amp": self.rc/self.Ql, "phase_off": self.theta0,
                  "e_delay": self.e_delay, "model":self.model}

        return params

    def run(self, debug=True):
        self._fit_circ_and_phase(debug)
        params = self.extract_params()
        return TransResult(self.coordinates, self.data, params)


class TransResult(ResonatorResult):
    def __init__(self, freq, data, params):
        super().__init__(freq, data, params)



if __name__ == '__main__':
    # test params
    f_list = np.linspace(5.02e9, 5.08e9, 1001)
    f0_ = 5.05e9
    Ql_ = 1e3
    Qc_m_ = 3e3
    amp_ = 1e-3
    phase_off_ = np.pi / 2.33 * 3
    e_delay_ = -2e-9 * 2
    phi_ = np.pi / 10 * 3

    # ----------- generate test hanger data
    data = hanger_func(f_list, f0_, Ql_, Qc_m_, amp_, phase_off_, e_delay_, phi_)
    data += (np.random.rand(len(data)) + 1j * np.random.rand(len(data))) * amp_ *0.05
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(f_list, np.abs(data))
    ax[1].plot(f_list, np.unwrap(np.angle(data)))
    # ----------------- fit hanger
    hf = HangerFit(f_list, data).run()
    hf.plot()
    hf.print()
    #
    # # # -------------- compare with old code
    # from QuDataProcessing.fitter.cavity_functions_hanger import CavHanger
    # hf_old = CavHanger(f_list, data).run()
    # hf_old.plot()
    # hf_old.print()



    # # --------------- generate reflection test data
    # data = refl_func(f_list, f0_, Ql_, Qc_m_, amp_, phase_off_, e_delay_)
    # data += (np.random.rand(len(data)) + 1j * np.random.rand(len(data))) * amp_ *0.03
    # plt.figure()
    # plt.plot(data.real, data.imag, ".")
    # # # -------------- fit reflection
    # rf = ReflFit(f_list, data).run(0)
    # rf.plot()
    # rf.print()

    # ---------- compare with old code
    # from QuDataProcessing.fitter.cavity_functions import CavReflection
    # rf_old = CavReflection(f_list, data).run()
    # rf_old.plot()


    # # --------------- generate transmission test data
    # data = trans_func(f_list, f0_, Ql_, amp_, phase_off_, e_delay_)
    # data += (np.random.rand(len(data)) + 1j * np.random.rand(len(data))) * amp_ * Qc_m_ *0.02
    # fig, ax = plt.subplots(1, 2)
    # ax[0].plot(f_list, np.abs(data))
    # ax[1].plot(f_list, np.unwrap(np.angle(data)))
    # # # -------------- fit reflection
    # tf = TransFit(f_list, data).run()
    # tf.plot()
    # tf.print()
    # # ---------- compare with old code




    # ------------- test find circ 0 ------------------------------
    # x = data.real
    # y = data.imag
    #
    # z = x ** 2 + y ** 2
    # v = np.array([z, x, y, np.ones_like(x)])
    # M = np.matmul(v, v.T)
    # B = np.zeros((4, 4))
    # B[0, 3] = B[3, 0] = -2
    #
    # # to solve for eta_, Newton iteration is overkill and not efficient
    # # M - eta_ * B is a quadratic function (since B only have two non-zero elements),
    # # we just need to find the coefficients for this quadratic function
    # c2 = 4 * (M[1, 2] * M[2, 1] - M[1, 1] * M[2, 2])
    # c0 = linalg.det(M)
    # c1 = linalg.det(M - B) - c2 - c0
    #
    # # minimum positive root for the quadratic function
    # c_dis = np.sqrt(c1 ** 2 - 4 * c2 * c0)
    # v1, v2 = (-c1 + c_dis) / 2 / c2, (-c1 - c_dis) / 2 / c2
    # print(v1, v2)
    # eta_ = min([v_ for v_ in [v1, v2] if v_ > 0], default=None)
    #
    #
    #
    # def solve_eta(eta):
    #     return linalg.det(M - eta * B)
    # eta_ = newton(solve_eta, 0)
    #
    #
    # print(eta_, linalg.det(M - eta_ * B), linalg.det(M))
    #
    # a, b, c, d = linalg.null_space(M - eta_ * B)[:, 0]
    # x0 = - b / 2 / a
    # y0 = - c / 2 / a
    # r0 = np.sqrt(b ** 2 + c ** 2 - 4 * a * d) / 2 / abs(a)
    # plt.figure()
    # plt.plot(data.real, data.imag, ".-")
    # theta_ = np.linspace(0, TWOPI, 1001)
    # plt.plot(x0 + r0*np.cos(theta_), y0 + r0*np.sin(theta_))


    # # -------------- test fit e_delay --------------------------
    # def circ_res(tau):
    #     rot_data = np.exp(1j * TWOPI * f_list * tau) * data
    #     x, y = rot_data.real, rot_data.imag
    #     x0, y0, r0 = fit_circle(x, y)
    #     return np.sum((r0 ** 2 - (x - x0) ** 2 - (y - y0) ** 2) ** 2)
    #
    #
    # tl = np.linspace(0e-9, 20e-9, 501)
    # plt.figure()
    # plt.plot(tl, [circ_res(t_) for t_ in tl])
    #
    # tau0 = guess_e_delay(f_list, data)
    # # tau0 = e_delay
    # # tau_fit = minimize(circ_res, tau0, method="Nelder-Mead").x[0]
    # # tau_fit = minimize(circ_res, tau0, method="Powell").x[0]
    # tau_fit = least_squares(circ_res, tau0, method="lm").x[0]
    # print(tau_fit, e_delay_)
    #
    # new_data = data * np.exp(1j * tau_fit * TWOPI * f_list)
    # plt.figure()
    # plt.plot(new_data.real, new_data.imag, ".")
    # fit_cir_new = fit_circle(new_data.real, new_data.imag)
    # theta_ = np.linspace(0, TWOPI, 1001)
    # plt.plot(fit_cir_new[0] + fit_cir_new[2] * np.cos(theta_), fit_cir_new[1] + fit_cir_new[2] * np.sin(theta_))

    # ------------- test fit phase ------------------------------
    # # translate circle data to origin
    # data_tr = new_data - (fit_cir_new[0] + 1j * fit_cir_new[1])
    # # phase fit
    # data_tr_phase = np.unwrap(np.angle(data_tr))
    # phase = data_tr_phase
    # freq = f_list
    # def phse_func(f, theta0, Ql, f0):
    #     theta = theta0 - 2 * np.arctan(2 * Ql * (f / f0 - 1))
    #     return theta
    #
    # model = Model(phse_func)
    # theta0_gue = (phase[0] + phase[-1]) / 2
    # f0_gue = freq[np.argmin(np.abs(phase - theta0_gue))]
    # Ql_gue = np.mean(freq) / (np.max(freq) - np.min(freq)) * 5
    # params = {}
    # params["theta0"] = Parameter("theta0", theta0_gue)
    # params["f0"] = Parameter("f0", f0_gue, min=np.min(freq), max=np.max(freq))
    # params["Ql"] = Parameter("Ql", Ql_gue, min=Ql_gue / 50, max=Ql_gue * 50)
    # fit_result = model.fit(phase, f=freq, **params)
    # fit_result.plot()

    #
    #
    #






    # plt.figure()
    # el = np.linspace(-10,10, 1001)
    # plt.plot(el, [linalg.det(M-e_*B) for e_ in el])
    # plt.plot(el, np.ones_like(el))
    # plt.plot(el, c2*el**2 + c1*el+c0)
