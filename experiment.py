import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal
from scipy.optimize import curve_fit

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


        

class OTdataset:
    '''Parent class for passive and active datasets, ideally should handle the calibration & the function generator calibration'''
    def __init__(self, fh, calibration_params=None):
        ''' 
        Args:
        fh: file handle for the dataset (NO TAILING SLASH) ex: 'My Drive/Altman Lab Articles and Protocols/UPLOADED DATA/name/date/bead1'
        optional:
        calibration_params: tuple of floats, (Sx, Sy, kx, ky, trapx, trapy) -> calibration parameters for the trap
        '''
        self.fh = fh
        if calibration_params is None: 
        # if no calibration parameters are given, calibrate the trap from the brownian data
            self.Sx, self.Sy, self.kx, self.ky, self.trapx, self.trapy = self._brownian_analysis() 
        else:
            self.Sx, self.Sy, self.kx, self.ky, self.trapx, self.trapy = calibration_params

    def _load_data(self, fh:str, return_power_spectrum:bool = False) -> tuple[np.array, np.array, np.array, np.array]:
        '''Load the data from file, concatenate it and return it as a tuple of arrays
        Optionally return the power spectrum of the data as well

        Args:
        fh: file handle
        return_power_spectrum: bool = False, whether to return the power spectrum of the data or not

        Returns:
        longdatax: np.array, x position data
        longdatay: np.array, y position data
        longdatasum: np.array, sum of x and y position data
        longdatafg: np.array, force data
        optional:
        powerspecx: np.array, power spectrum of x position data
        powerspecy: np.array, power spectrum of y position data
        '''
        def file_sort(s):
            return int(s.split('_')[-1]) # slow
        
        count = 0
        longdatax = []
        longdatay = []
        longdatasum = []
        longdatafg = []
        powerspecx = np.zeros(2501)
        powerspecy = np.zeros(2501)

        fhs = sorted(os.listdir(fh), key=file_sort)
        for fh in fhs:
            dim = openfile(fh)
            datax = dim[0::4]
            datay = dim[2::4]
            datafg = dim[3::4]
            datasum = dim[1::4]

            longdatax = np.append(longdatax, datax)
            longdatay = np.append(longdatay, datay)
            longdatasum = np.append(longdatasum, datasum)
            longdatafg = np.append(longdatafg, datafg)

            if return_power_spectrum:
                breakitup = 10
                for k in range(0, int(fs), int(fs/breakitup)):
                    dataxchunk = datax[k:int(k+(fs/breakitup))]
                    dataychunk = datay[k:int(k+(fs/breakitup))]
                    fx, Pxx_den = calcpowerspec2(dataxchunk)
                    powerspecx = powerspecx + Pxx_den
                    fy, Pyy_den = calcpowerspec2(dataychunk)
                    powerspecy = powerspecy + Pyy_den
                    count = count+1
        
        
        if return_power_spectrum:
            powerspecx = powerspecx/count
            powerspecy = powerspecy/count
            return longdatax, longdatay, longdatasum, longdatafg, fx, powerspecx, powerspecy
        
        return longdatax, longdatay, longdatasum, longdatafg
            

    def _brownian_analysis(self):
        '''Given the file handle in self, run load_data for brownian data, and return: 
        Sx, Sy: (V/m) -> mapping from voltage to position
        kx, ky: (N/m) -> spring constant of the trap for x and y
        trapx, trapy: (m) -> position of the center of the trap
        '''
        fh = self.fh[:self.fh.rfind('/')+1]
        ls_dir = os.listdir(fh)
        if 'brownian' in ls_dir:
            fh += 'brownian/brownian_'
        elif 'Brownian' in ls_dir:
            fh += 'Brownian/Brownian_'
        else:
            raise ValueError('No brownian data found in the folder')
        
        xb, yb, sumsignalb, fg, f, powerspecx, powerspecy = self._load_data(fh, return_power_spectrum=True)
        initial = 3 # starting frequency for the fit
        list = [i for i in range(len(f)) if f[i] > 1000] 
        final = list[0]

        # Fit the power spectrum to get the trap stiffness
        # x
        poptx, pcov = curve_fit(func, f[initial:final], powerspecx[initial:final], p0=[1e5, 1e-3])
        alphax = abs(poptx[1])

        # y
        popty, pcov = curve_fit(func, f[initial:final], powerspecy[initial:final], p0=[1e5, 1e-3])
        alphay = abs(popty[1])

        return poptx[0], popty[0], alphax, alphay, np.mean(xb), np.mean(yb)


    
    
    def see_calibration(self):
        '''See the calibration of the dataset'''
        pass

    def calibrate_function_gen(self, plots=['lin_fit']):
        '''Calibrate the function generator
        Args:
        plots: list of plots to show, default has 'lin_fit'
        '''
        plots = {}
        for plot_name in plots:
            plots[plot_name] = True
        

class passiveOT(OTdataset):
    '''Class for processing passive OT datasets'''
    def __init__(self, fh, subhandle):
        '''fh: file handle for where brownian folder can be found, subhandle: relative file handle of the passive dataset i.e. bead1p/'''
        super().__init__(fh)

class activeOT(OTdataset):
    '''Class for processing active OT datasets'''
    def __init__(self, fh, subhandle):
        '''fh: file handle for where brownian folder can be found, subhandle: relative file handle of the active dataset i.e. bead1/'''
        super().__init__(fh)

    