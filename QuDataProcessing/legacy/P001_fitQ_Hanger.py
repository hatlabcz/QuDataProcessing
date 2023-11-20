import numpy as np
import matplotlib.pyplot as plt
import inspect
from scipy.optimize import curve_fit
from P001_fitQ import getData, printFitQResult

FREQ_UNIT = {'GHz': 1e9,
             'MHz': 1e6,
             'KHz': 1e3,
             'Hz': 1.0
             }


def rounder(value):
    return "{:.4e}".format(value)


def hangerFuncMagAndPhase(omega, Qext, Qint, omega0, magBack, delta, phaseCorrect):
    x = (omega - omega0) / (omega0)
    S_21_up = Qext + 1j * Qext * Qint * (2 * x + 2 * delta / omega0)
    S_21_down = (Qint + Qext) + 2 * 1j * Qext * Qint * x
    S21 = magBack * (S_21_up / S_21_down) * np.exp(1j * (phaseCorrect))  # model by Kurtis Geerlings thesis
    mag = np.log10(np.abs(S21)) * 20
    phase = np.angle(S21)
    return (mag + 1j * phase).view(float)

def hangerFuncRealAndImag(omega, Qext, Qint, omega0, magBack, delta, phaseCorrect):
    x = (omega - omega0) / (omega0)
    S_21_up = Qext + 1j * Qext * Qint * (2 * x + 2 * delta / omega0)
    S_21_down = (Qint + Qext) + 2 * 1j * Qext * Qint * x
    S21 = magBack * (S_21_up / S_21_down) * np.exp(1j * (phaseCorrect))  # model by Kurtis Geerlings thesis
    return S21.view(float)


def fit(freq, real, imag, mag, phase, Qguess=(2e5, 1e5), bounds=None, f0Guess=None, magBackGuess=None, delta=10e6,
        phaseGuess=0):
    if f0Guess == None:
        f0Guess = freq[int(np.floor(np.size(freq) / 2))]  # dumb guess of "it's probably in the middle"
    # #smart guess of "it's probably the lowest point"
    print("Guess freq: " + str(f0Guess / (2 * np.pi * 1e9)))

    lin = 10 ** (mag / 20.0)
    if magBackGuess == None:
        magBackGuess = np.average(lin[:int(len(freq) / 5)])
    QextGuess = Qguess[0]
    QintGuess = Qguess[1]
    if bounds == None:
        bounds = ([QextGuess / 20, QintGuess / 20, freq[0], magBackGuess / 10, -1e9, -np.pi],
                  # bounds can be made tighter to allow better convergences
                  [QextGuess * 20, QintGuess * 20, freq[-1], magBackGuess * 10, 1e9, np.pi])

    target_func = hangerFuncMagAndPhase
    data_to_fit = (mag + 1j * phase).view(float)
    # data_to_fit = (real  + 1j * imag).view(float)
    popt, pcov = curve_fit(target_func, freq, data_to_fit,
                           p0=(QextGuess, QintGuess, f0Guess, magBackGuess, delta, phaseGuess),
                           bounds=bounds,
                           maxfev=1e4, ftol=2.3e-16, xtol=2.3e-16)

    return popt, pcov

def plotRes(freq, real, imag, mag, phase, popt):
    xdata = freq/ (2 * np.pi)
    magRes = hangerFuncMagAndPhase(freq, *popt)[::2]
    faseRes = hangerFuncMagAndPhase(freq, *popt)[1::2]
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.title('mag (db)')
    plt.plot(xdata, mag, '.')
    plt.plot(xdata, magRes)
    plt.subplot(1, 2, 2)
    plt.title('phase')
    plt.plot(xdata, phase, '.')
    plt.plot(xdata, faseRes)
    plt.show()


# def plotResRealImag(freq, real, imag, mag, phase, popt):
#     xdata = freq/ (2 * np.pi)
#     magRes = hangerFuncRealAndImag(freq, *popt)[::2]
#     faseRes = hangerFuncRealAndImag(freq, *popt)[1::2]
#     plt.figure(figsize=(12, 5))
#     plt.subplot(1, 2, 1)
#     plt.title('real')
#     plt.plot(xdata, real, '.')
#     plt.plot(xdata, magRes)
#     plt.subplot(1, 2, 2)
#     plt.title('imag')
#     plt.plot(xdata, imag, '.')
#     plt.plot(xdata, faseRes)
#     plt.show()


if __name__ == '__main__':
    # filepath = easygui.fileopenbox()
    # filepath = r'L:\Data\HouckQubit\HouckSample\Cav\\Q1Cav-20dBm'
    filepath = r'C:\Users\hatlab-pxie2\Downloads\\01-06-2022_Si-NbTiN_6388777541_1601pts_500avgF_EDelay_50_PhOffset_-180_power_w_atten_-60'
    (freq, real, imag, mag, phase) = getData(filepath, plot_data=0)
    ltrim = 0  # trims the data if needed
    rtrim = 1  # keep this value greater than 1
    freq = freq[ltrim:-rtrim]
    real = real[ltrim:-rtrim]
    imag = imag[ltrim:-rtrim]
    mag = mag[ltrim:-rtrim]
    phase = phase[ltrim:-rtrim]

    popt, pcov = fit(freq, real, imag, mag, phase, Qguess=(1e4, 3e6), delta=10e6)  # (ext, int)

    printFitQResult(popt,pcov)

    plotRes(freq, real, imag, mag, phase, popt)