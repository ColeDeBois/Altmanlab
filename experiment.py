import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
plt.style.use('default')
import os
from scipy import signal
from scipy.optimize import curve_fit
import scipy.fft

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
        fh: file handle for the dataset !!NO TAILING SLASH!! ex: 'My Drive/Altman Lab Articles and Protocols/UPLOADED DATA/name/date/bead1'
        optional:
        calibration_params: tuple of floats, (Sx, Sy, kx, ky, trapx, trapy) -> calibration parameters for the trap
        '''
        self.fh = fh
        if calibration_params is None: 
        # if no calibration parameters are given, calibrate the trap from the brownian data
            self.Sx, self.Sy, self.kx, self.ky, self.trapx, self.trapy = self._brownian_analysis(visualize=True) 
        else:
            self.Sx, self.Sy, self.kx, self.ky, self.trapx, self.trapy = calibration_params
        print('Calibration parameters:')
        print('Sx: ', self.Sx)
        print('Sy: ', self.Sy)
        print('kx: ', self.kx)
        print('ky: ', self.ky)
        print('trapx: ', self.trapx)
        print('trapy: ', self.trapy)

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
        fhs = fhs[:-1] # remove the last file, as this causes errors
        for i, sh in enumerate(fhs):
            dim = openfile(fh+sh)
            datax = dim[0::4]
            datay = dim[2::4]
            datafg = dim[3::4]
            datasum = dim[1::4]
            for i in range(len(datax)):
                longdatax.append(datax[i])
                longdatay.append(datay[i])
                longdatasum.append(datasum[i])
                longdatafg.append(datafg[i])

            if return_power_spectrum:
                breakitup = 10
                for k in range(0, int(fs), int(fs/breakitup)):
                    dataxchunk = datax[k:int(k+(fs/breakitup))]
                    dataychunk = datay[k:int(k+(fs/breakitup))]
                    fx, Pxx_den = calcpowerspec2(dataxchunk)
                    powerspecx += Pxx_den
                    fy, Pyy_den = calcpowerspec2(dataychunk)
                    powerspecy += Pyy_den
                    count = count+1
        
        
        if return_power_spectrum:
            powerspecx = powerspecx/count
            powerspecy = powerspecy/count
            return np.array(longdatax), np.array(longdatay), np.array(longdatasum), np.array(longdatafg), fx, powerspecx, powerspecy
        
        return np.array(longdatax), np.array(longdatay), np.array(longdatasum), np.array(longdatafg)
            

    def _brownian_analysis(self, visualize=False):
        '''Given the file handle in self, run load_data for brownian data, and return: 
        Sx, Sy: (V/m) -> mapping from voltage to position
        kx, ky: (N/m) -> spring constant of the trap for x and y
        trapx, trapy: (m) -> position of the center of the trap
        '''
        fh = self.fh[:self.fh.rfind('/')+1]
        ls_dir = os.listdir(fh)
        if 'brownian' in ls_dir:
            fh += 'brownian/'
        elif 'Brownian' in ls_dir:
            fh += 'Brownian/'
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
        if visualize:
            fig=plt.figure(figsize=(10, 10), tight_layout=True)
            fig.suptitle('OT Calibration')
            xplot = fig.add_subplot(221)
            xplot.set_title('Power Spectrum of X')
            xplot.loglog(f, powerspecx, 'k.')
            xplot.loglog(f[initial:final], func(f[initial:final], *poptx), 'r-')
            xplot.set_xlabel('Frequency (Hz)')
            xplot.set_ylabel('Power Spectrum ($V^2$/Hz)')
            
            yplot = fig.add_subplot(222)
            yplot.set_title('Power Spectrum of Y')
            yplot.loglog(f, powerspecy, 'k.')
            yplot.loglog(f[initial:final], func(f[initial:final], *popty), 'r-')
            yplot.set_xlabel('Frequency (Hz)')
            yplot.set_ylabel('Power Spectrum ($V^2$/Hz)')
            
            fig2=plt.figure(figsize=(10, 5))
            fig2.suptitle('Position')
            t = np.arange(len(xb))*dt
            xpos=fig2.add_subplot(311)
            xpos.plot(t,xb,'k')
            xpos.set_ylabel('x[V]')
            ypos=fig2.add_subplot(312)
            ypos.plot(t,yb,'k')
            ypos.set_ylabel('y[V]')
            ypos.set_xlabel('Time [s]')
            sumsignal=fig2.add_subplot(313)
            sumsignal.plot(t,sumsignalb,'k')
            sumsignal.set_ylabel('sum[V]')
            sumsignal.set_xlabel('Time [s]')
            plt.plot()



        Sx=poptx[0]
        Sy=popty[0]
        return Sx, Sy, alphax, alphay, np.mean(xb)/Sx, np.mean(yb)/Sy #TODO: check if this is correct for trapx & trapy


    
    
    def see_calibration(self):
        '''See the calibration of the trap for the dataset'''
        pass

    def calibrate_function_gen(self, show_plots=['lin_fit']):
        '''Calibrate the function generator
        Args:
        plots: list of plots to show, default has only the 'lin_fit'

        Returns:
        None, but sets self.a and self.b to the calibration parameters, which is needed for active analysis
        '''
        plots = {'position': False, 'signal_jumps': False, 'lin_fit': False}
        for plot_name in show_plots:
            plots[plot_name] = True
        
        # Load the data
        fh = self.fh[:self.fh.rfind('/')+1]
        ls_dir = os.listdir(fh)
        if 'functiong' in ls_dir:
            fh += 'functiong/'
        elif 'FunctionG' in ls_dir:
            fh += 'FunctionG/'
        else:
            raise ValueError('No function generator data found in the folder')
        x, y, sumsignal, fg = self._load_data(fh)
        t = np.arange(len(x))*dt
        x = x/self.Sx #convert to m
        y = y/self.Sy #convert to m

        # Plot the position data
        if plots['position']:
            fig = plt.figure(figsize=(10, 5), tight_layout=True)
            fig.suptitle('Position vs Function Generator')
            ax = fig.add_subplot(211)
            ax.set_title('Bead Position (nm)')
            ax.plot(t, y*1e9, 'k')
            ax.set_ylabel('Position (m)')
            plt.xlabel('Time (s)')

            ax1 = fig.add_subplot(212)
            ax1.set_title('Function Generator Signal (V)')
            ax1.plot(t, fg, 'k')
            ax1.set_ylabel('Signal (V)')
        
        # Plot the function generator signal jumps
        absdiff = np.abs(np.diff(fg))
        jumpy_idxs = absdiff > (np.mean(absdiff)*5)
        
        if plots['signal_jumps']:
            fig = plt.figure(figsize=(10, 5), tight_layout=True)
            fig.suptitle('Function Generator Signal Jumps')
            ax = fig.add_subplot(211)
            ax.set_title('Highlighted Non-jump Signal')
            ax.plot(t[1:], fg[1:], 'k')
            ax.plot(t[1:][jumpy_idxs], fg[1:][jumpy_idxs], 'r.')
            ax.set_ylabel('Signal (V)')
            plt.xlabel('Time (s)')
            bx = fig.add_subplot(212)
            bx.set_title('Absolute Differences')
            bx.plot(t[1:], absdiff, 'k,')
            bx.set_ylabel('Signal Difference (V)')
        
        # Reduce signal jumps to a single point i.e. throw out the in between points
        idx = []
        previous: bool = jumpy_idxs[0]
        for i, current in enumerate(jumpy_idxs):
            if not current and previous: # Last point was a jump, and this is not a jump
                idx.append(i-1) #Marking the end of jumps for exclusive indexing
            previous = current
        
        stdfg = []
        for i in range(len(idx)-1):
            rangeval = (t[idx[i]] < t) * (t < t[idx[i+1]])
            stdfg.append(np.std(np.abs(fg[rangeval])))
        stdfg = np.array(stdfg)
        
        avg_fg = []
        avg_y = []
        threshold = 1.1*np.mean(stdfg) # setting the threshold to the mean of the std of the signal
        for i in range(len(stdfg)-1):
            if stdfg[i] < threshold: # ensure the fg is stable (i.e. we're not in a jump)
                rangeval = (t[idx[i] + int(fs/2 * .75)] < t) * (t < t[idx[i] + int(fs/2 * .95)]) #getting the last 0.2 seconds of the signal
                meanfg = np.mean(fg[rangeval])
                meany = np.mean(y[rangeval])
                print(f'meanfg: {meanfg}, meany: {meany}')
                avg_fg.append(meanfg)
                avg_y.append(meany)
        
        avg_fg = np.array(avg_fg)
        avg_y = np.array(avg_y)

        # Fit the avg fg to the avg y in our linear region
        plt.figure()
        upper_bound = 150*1e-9 + self.trapy
        lower_bound = -200*1e-9 + self.trapy
        inrange = (lower_bound < avg_y) * (avg_y < upper_bound)
        a, b = np.polyfit(avg_fg[inrange], 1e9*avg_y[inrange], 1)
        plt.plot(avg_fg, 1e9*avg_y, 'k,')
        plt.plot(avg_fg[inrange], a* avg_fg[inrange] + b, 'g-')
        plt.plot(avg_fg[inrange], 1e9*avg_y[inrange], 'r.', alpha=0.5)
        self.a=a
        self.b=b

                

class Passive_Analysis(OTdataset):
    '''Class for processing passive OT datasets'''
    def __init__(self, fh, subhandle):
        '''fh: file handle for where brownian folder can be found, subhandle: relative file handle of the passive dataset i.e. bead1p/'''
        super().__init__(fh)
        
        # Load the data
        self.fh = fh + subhandle
        self.x, self.y, self.sumsignal, self.fg = self._load_data(self.fh)
        self.t = np.arange(len(self.x))*dt


        
    

class Active_Analysis(OTdataset):
    '''Class for processing active OT datasets'''
    def __init__(self, fh, calibration_params=None):
        '''fh: file handle of the active dataset i.e. bead1 !!!NO TAILING SLASH!!! '''
        super().__init__(fh, calibration_params)

        # Load the data
        self.x, self.y, self.sumsignal, self.fg = self._load_data(self.fh+'/')
        self.x/self.Sx - self.trapx, self.y/self.Sy - self.trapy
        self.t = np.arange(len(self.x))*dt
    
    def kowalski__analysis(self):
        '''Run Kowalski analysis on the active data'''
        x,y,fg,t = self.x, self.y, self.fg, self.t #just for ease
        # Calculate the power spectrum of the data
        frequencies = []
        for i in range(0, len(x), fs):
            Y = scipy.fft.fft(fg[i:i+fs]) # indexing one second of data for this (aka the entire bead_{i} file)
            N = len(Y)
            n = np.arange(N)
            T = N/fs
            freq = n/T
            # filter out the frequencies that are out of our working range
            working_range: np.ndarray[np.bool_] = (freq>0.1)*(freq<20000) 
            freq = freq[working_range]
            Y = np.abs(Y[working_range])
            for f in freq[Y == max(Y)]:
                frequencies.append(f)
            
        def sinefit(t, A, C, D, omega):
            return np.abs(A) * np.sin(omega*t - C)+D
        def sinefit2(freqoscil):
            def func(t,*p):
                return np.abs(p[0]) * np.sin(freqoscil*t - p[1])+p[2]
            return func
        
        freqlist = []
        fnot = []
        ynot = []
        delta = []

        for i in range(len(frequencies)):
            if frequencies[i] < 30:
                Nfit = 10 # number of oscillations to fit at the lower range (<30Hz)
            else:
                Nfit = 100
            T = 1/frequencies[i] #period
            x = self.x[i*fs:(i+int(Nfit*T))*fs] #we are indexing the data in seconds needed for Nfit oscillations
            y = self.y[i*fs:(i+int(Nfit*T))*fs]
            fg = self.fg[i*fs:(i+int(Nfit*T))*fs]
            t = np.arange(len(x))*dt

            plt.plot(t, fg, 'k')
            
            params, params_cov = scipy.optimize.curve_fit(sinefit, t, fg, p0=[np.std(fg)*np.sqrt(2),np.pi, 0, 2*np.pi*frequencies[i]])
            phase = params[1]
            fg_freq = params[3]

            plt.plot(t, sinefit(t, *params), 'r-')
