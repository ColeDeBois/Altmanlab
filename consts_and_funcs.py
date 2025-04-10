import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.fft
import os
from scipy import signal
plt.style.use('default')


fs = 50000  # data collection rate
dt = 1/fs
kBT = 4.11e-21     # room temp (J)
#eta = 0.001  # Pa s, viscosity of water
eta = 0.002548 # Pa s, viscosity of BSA solution
r = (2.0e-6)/2      # radius of bead
D = kBT /(6*np.pi*eta*r)    # diffusion constant
beta = 6*np.pi*eta*r

def openfile(filename):   # open files
        fid = open(filename, 'rb')
        dim = np.fromfile(fid, dtype='>d') # big-endian (google it lol) decimal
        fid.close()
        return dim

def calcpowerspec2(data):    # Calculate power spectrum
    f, Pxx_den = signal.welch(data, fs,  nperseg=int(len(data)), window='hamming')
    return f, Pxx_den

def func(x, a, b):    # fitting function for the PSD
    return a**2*(D/((np.pi)**2)) / ( (b/(2*np.pi*beta))**2 + x**2) # from eqn 10 Berg-Sorensen et al. 2004
       # a - Sensitivity - units of V/m
       # b - Stiffness - N/m

def msd_fft(x, y):      # calculate the mean squared displacement
    # Organize the data into a 2-column array where each row contains an x and y position
    r = np.ones((int(len(x)), 2))
    for i in range(0,int(len(x))):
        values = [x[i], y[i]]
        r[i] =  values

def calcmsd2(x, y, analysislength, msdlength):
            dt = 1/50000
            fs = 50000
        # Select data to analyze MSD and fourier transform
            xdata = x[0:int(analysislength/dt)]
            ydata = y[0:int(analysislength/dt)]


            sos = signal.butter(10, 1, 'hp', fs=50000, output='sos')
            xdata = signal.sosfilt(sos, xdata)
            ydata = signal.sosfilt(sos, ydata)


            msdvalues = msd_fft(xdata, ydata)
            tautemp = msdvalues.tau
            msdtemp = msdvalues.msd
            msdvalues.tau = tautemp[0:int(msdlength/dt)]
            msdvalues.msd = msdtemp[0:int(msdlength/dt)]

            plt.figure()
            plt.loglog(msdvalues.tau[1:int(msdlength/(dt))], msdvalues.msd[1:int(msdlength/(dt))], 'b.', label = 'Bead in the Optical Trap')
            plt.figure()
            msddata = np.array(msdvalues.msd[1:int(msdlength/dt)]) / (1e18)
            N = len(msddata)
            T = 1/fs
            yf = scipy.fft.fft(msddata)
            xf = scipy.fft.fftfreq(N, T)[:N//2]

            print(len(xf))
            print(len(yf))
            plt.plot(xf[100:1500], 2.0/N * np.abs(yf[100:1500]))
            plt.xlabel('frequency (Hz)')
            plt.title('Fourier transform of the MSD')
            plt.show()
            return msdvalues