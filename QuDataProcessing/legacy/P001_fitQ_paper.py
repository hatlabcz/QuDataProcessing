import numpy as np
import matplotlib.pyplot as plt
import csv
import h5py
import inspect
from scipy.optimize import curve_fit

# import easygui

FREQ_UNIT = {'GHz': 1e9,
             'MHz': 1e6,
             'KHz': 1e3,
             'Hz': 1.0
             }


def rounder(value, digit=5):
    return f"{value:.{digit}e}"


def reflectionFunc(freq, Qext, Qint, f0, magBack, phaseCorrect):
    omega0 = f0
    delta = freq - omega0
    S_11_up = 1.0 / (1j * delta * (2 + delta / omega0) / (1 + delta / omega0) + omega0 / Qint) - Qext / omega0
    S_11_down = 1.0 / (1j * delta * (2 + delta / omega0) / (1 + delta / omega0) + omega0 / Qint) + Qext / omega0
    S11 = magBack * (S_11_up / S_11_down) * np.exp(1j * (phaseCorrect))
    realPart = np.real(S11)
    imagPart = np.imag(S11)

    return (realPart + 1j * imagPart).view(np.float)
    # return realPart 
    # return imagPart 

def reflectionFunc_semi(freq, Qext, Qint, f0, magBack, phaseCorrect):
    omega0 = f0
    delta = freq - omega0
    S_11_up = (omega0/Qint - omega0/Qext)/2 - 1j * delta
    S_11_down = (omega0/Qint + omega0/Qext)/2 - 1j * delta
    S11 = magBack * (S_11_up / S_11_down) * np.exp(1j * (phaseCorrect))

    # mag = np.abs(S11)
    # phase = np.angle(S11)
    realPart = np.real(S11)
    imagPart = np.imag(S11)
    return (realPart + 1j * imagPart).view(np.float)

def reflectionFunc_paper(freq, Qext, Qint, f0, magBack, phaseCorrect):
    omega0 = f0
    delta = freq - omega0
    S_11_up = 1 - Qint/Qext + 1j * 2 * Qint * delta / omega0
    S_11_down = 1 + Qint/Qext + 1j * 2 * Qint * delta / omega0
    S11 = magBack * (S_11_up / S_11_down) * np.exp(1j * (phaseCorrect))

    # mag = np.abs(S11)
    # phase = np.angle(S11)
    realPart = np.real(S11)
    imagPart = np.imag(S11)
    return (realPart + 1j * imagPart).view(np.float)


def reflectionFunc_phaseOnly(freq, Qext, Qint, f0, phaseCorrect, w):
    omega0 = f0
    delta = freq - omega0
    S_11_up = 1 - Qint/Qext + 1j * 2 * Qint * delta / omega0
    S_11_down = 1 + Qint/Qext + 1j * 2 * Qint * delta / omega0
    S11 = (S_11_up / S_11_down)

    realPart = np.real(S11)
    imagPart = np.imag(S11)
    return np.unwrap(np.angle(realPart + 1j * imagPart).view(np.float)) + w * delta + phaseCorrect


def reflectionFunc_re(freq, Qext, Qint, f0, magBack, phaseCorrect):
    return reflectionFunc(freq, Qext, Qint, f0, magBack, phaseCorrect)[::2]


def getData(filename, method='hfss', freq_unit='GHz', plot_data=1):
    if method == 'hfss':
        """The csv file must be inthe format of:
            freq  mag(dB)  phase(cang_deg)  
        """
        with open(filename) as csvfile:
            csvData = list(csv.reader(csvfile))
            csvData.pop(0)  # Remove the header
            data = np.zeros((len(csvData[0]), len(csvData)))
            for x in range(len(csvData)):
                for y in range(len(csvData[0])):
                    data[y][x] = csvData[x][y]

        freq = data[0] * 2 * np.pi * FREQ_UNIT[freq_unit]  # omega
        phase = np.array(data[2]) / 180. * np.pi
        mag = data[1]
        lin = 10 ** (mag / 20.0)

    elif method == 'vna':
        f = h5py.File(filename, 'r')
        freq = f['VNA Frequency (Hz)'][()] * 2 * np.pi
        phase = f['Phase (deg)'][()] / 180. * np.pi
        mag = f['Power (dB)'][()]
        lin = 10 ** (mag / 20.0)
        f.close()

    elif method == 'vna_old':
        f = h5py.File(filename, 'r')
        freq = f['Freq'][()] * 2 * np.pi
        phase = f['S21'][()][1] / 180 * np.pi
        mag = f['S21'][()][0]
        lin = 10 ** (mag / 20.0)
        f.close()

    else:
        raise NotImplementedError('method not supported')

    real = lin * np.cos(phase)
    imag = lin * np.sin(phase)

    if plot_data:
        plt.figure('mag')
        plt.plot(freq / 2 / np.pi, mag)
        plt.figure('phase')
        plt.plot(freq / 2 / np.pi, phase)

    return (freq, real, imag, mag, phase)

    # if method == 'vna':  
    #     f = h5py.File(filename,'r')
    #     freq = f['VNA Frequency (Hz)'][()]
    #     phase = f['Phase (deg)'][()] / 180. * np.pi
    #     lin = 10**(f['Power (dB)'][()] / 20.0)
    # if method == 'vna_old': 
    #     f = h5py.File(filename,'r')
    #     freq = f['Freq'][()]
    #     phase = f['S21'][()][0] / 180. * np.pi
    #     lin = 10**(f['S21'][()][1] / 20.0)


def fit(freq, real, imag, mag, phase, Qguess=(2e3, 1e3), real_only=0):
    # f0Guess = 8.5596e9
    # f0Guess = freq[np.argmin(mag)]  # smart guess of "it's probably the lowest point"
    f0Guess = freq[int(np.floor(np.size(freq)/2))] #dumb guess of "it's probably in the middle"
    lin = 10 ** (mag / 20.0)
    magBackGuess = np.average(lin[:int(len(freq) / 5)])
    QextGuess = Qguess[0]
    QintGuess = Qguess[1]
    # bounds=([QextGuess / 100, QintGuess / 100.0, f0Guess - 10.1, magBackGuess / 10.0, -2 * np.pi],
    #         [QextGuess * 100, QintGuess * 100.0, f0Guess + 10.1, magBackGuess * 10.0, 2 * np.pi])

    bounds = ([QextGuess / 5, QintGuess / 2, f0Guess - 1e5 , magBackGuess / 1.1, - np.pi],
              [QextGuess * 5, QintGuess * 2, f0Guess + 1e5, magBackGuess * 1.1, np.pi])

    # target_func = reflectionFunc
    target_func = reflectionFunc_paper
    data_to_fit = (real + 1j * imag).view(np.float)
    if real_only:
        target_func = reflectionFunc_re
        data_to_fit = real
    popt, pcov = curve_fit(target_func, freq, data_to_fit,
                           p0=(QextGuess, QintGuess, f0Guess, magBackGuess, 0),
                           bounds=bounds,
                           maxfev=1e10, ftol=2.3e-20, xtol=2.3e-20)

    return popt, pcov


def fitPhaseOnly(freq, phase, Qguess=(2e3, 1e3), wGuess=0):
    # f0Guess = 8.5596e9
    # f0Guess = freq[np.argmin(mag)]  # smart guess of "it's probably the lowest point"

    f0Guess = freq[int(np.floor(np.average(np.where(abs(phase - np.average(phase))<0.2))))] #dumb guess of "it's probably in the middle"
    QextGuess = Qguess[0]
    QintGuess = Qguess[1]
    # bounds=([QextGuess / 100, QintGuess / 100.0, f0Guess - 10.1, magBackGuess / 10.0, -2 * np.pi],
    #         [QextGuess * 100, QintGuess * 100.0, f0Guess + 10.1, magBackGuess * 10.0, 2 * np.pi])

    bounds = ([QextGuess / 10, QintGuess / 10, f0Guess - 2e6, - 2 * np.pi, wGuess-1e-6],
              [QextGuess * 10, QintGuess * 10, f0Guess + 2e6, 2 * np.pi, wGuess+1e-6])

    # target_func = reflectionFunc
    target_func = reflectionFunc_phaseOnly
    data_to_fit = phase

    popt, pcov = curve_fit(target_func, freq, data_to_fit,
                           p0=(QextGuess, QintGuess, f0Guess, 0, wGuess),
                           bounds=bounds,
                           maxfev=1e10, ftol=2.3e-20, xtol=2.3e-20)

    return popt, pcov


def plotRes(freq, real, imag, mag, phase, popt):
    xdata = freq / (2 * np.pi)
    realRes = reflectionFunc_paper(freq, *popt)[::2]
    imagRes = reflectionFunc_paper(freq, *popt)[1::2]
    # realRes = reflectionFunc(freq, *popt)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.title('real')
    plt.plot(xdata, real, '.')
    plt.plot(xdata, realRes)
    plt.subplot(1, 2, 2)
    plt.title('imag')
    plt.plot(xdata, imag, '.')
    plt.plot(xdata, imagRes)
    plt.show()


def plotPhase(freq, phase, popt):
    xdata = freq / (2 * np.pi)
    phaseRes = reflectionFunc_phaseOnly(freq, *popt)
    plt.figure('phase')
    plt.title('phase')
    plt.plot(xdata, phase, '.')
    plt.plot(xdata, phaseRes)

def printFitQResult(popt,pcov):
    Qext = popt[0]
    Qint = popt[1]
    Qtot = popt[0] * popt[1] / (popt[0] + popt[1])
    freq_ = popt[2] / 2 / np.pi
    print(f'f (Hz): {rounder(freq_, 9)}+-{rounder(np.sqrt(pcov[2,2])/2/np.pi, 9)}')
    print(f'Qext: {rounder(Qext, 9)}+-{rounder(np.sqrt(pcov[0,0]), 9)}')
    print(f'Qint: {rounder(Qint, 9)}+-{rounder(np.sqrt(pcov[1,1]), 9)}')
    print('Q_tot: ', rounder(Qtot, 9))
    print('T1 (s):', rounder(Qtot / freq_ / 2 / np.pi, 9), '\nMaxT1 (s):', rounder(Qint / freq_ / 2 / np.pi, 9))
    print('kappa/2Pi: ', freq_ / Qtot / 1e6, 'MHz')
    return freq_ / Qtot

if __name__ == '__main__':
    # filepath = easygui.fileopenbox()
    plt.close('all')
    filepath = r'L:\Data\WISPE3D\Modes\20210809\CavModes\Cav'
    (freq, real, imag, mag, phase) = getData(filepath, method="vna_old", plot_data=1)
    trim = 1
    freq = freq[trim:-trim]
    real = real[trim:-trim]
    imag = imag[trim:-trim]
    mag = mag[trim:-trim]
    phase = phase[trim:-trim]

    popt, pcov = fit(freq, real, imag, mag, phase, Qguess=(1e4, 1e4))  # (ext, int)

    print(f'f (Hz): {rounder(popt[2] / 2 / np.pi)}')
    fitting_params = list(inspect.signature(reflectionFunc).parameters.keys())[1:]
    for i in range(2):
        print(f'{fitting_params[i]}: {rounder(popt[i])} +- {rounder(np.sqrt(pcov[i, i]))}')
    Qtot = popt[0] * popt[1] / (popt[0] + popt[1])
    print('Q_tot: ', rounder(Qtot), '\nT1 (s):', rounder(Qtot / popt[2]))
    print('kappa: ', (popt[2] / 2 / np.pi) / popt[0] / 1e6, 'MHz')
    plotRes(freq, real, imag, mag, phase, popt)

