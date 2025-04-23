'''the following code is for analyzing data from the optical trap, including passive and active analysis'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize
from scipy.optimize import curve_fit
import scipy.fft
import os
from scipy import signal
plt.style.use('default')


fs = 50000  # data collection rate
dt = 1/fs
kBT = 4.11e-21     # room temp (J)
#eta = 0.001  # Pa s, viscosity of water
eta = 0.002548 # Pa s, viscosity of BSA solution
eta = .001856 # viscosity of 20% iodixinol
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

def autocorrFFT(x):     # used by the program that calculates the meansquared displacement
    N=len(x)
    F = np.fft.fft(x, n=2*N)  #2*N because of zero-padding
    PSD = F * F.conjugate()
    res = np.fft.ifft(PSD)
    res= (res[:N]).real   #now we have the autocorrelation in convention B
    n=N*np.ones(N)-np.arange(0,N) #divide res(m) by (N-m)
    return res/n #this is the autocorrelation in convention A

def msd_fft(x, y):      # calculate the mean squared displacement
    # Organize the data into a 2-column array where each row contains an x and y position
    r = np.ones((int(len(x)), 2))
    for i in range(0,int(len(x))):
        values = [x[i], y[i]]
        r[i] =  values

    # calculate the MSD using the FFT
    N=len(r)
    D=np.square(r).sum(axis=1)
    D=np.append(D,0)
    S2=sum([autocorrFFT(r[:, i]) for i in range(r.shape[1])])
    Q=2*D.sum()
    S1=np.zeros(N)
    for m in range(N):
        Q=Q-D[m-1]-D[N-m]
        S1[m]=Q/(N-m)
    msdval = S1-2*S2
    tau = np.multiply(dt, [i for i in range(0, len(msdval))])
    msdvalues = pd.DataFrame({'tau': tau, 'msd': msdval})
    return msdvalues

def calcmsd2(x, y, analysislength, msdlength):
            dt = 1/50000
            fs = 50000
        # Select data to analyze MSD and fourier transform
            xdata = x[0:int(analysislength/dt)]
            ydata = y[0:int(analysislength/dt)]


            # sos = signal.butter(10, 1, 'hp', fs=50000, output='sos')
            # xdata = signal.sosfilt(sos, xdata)
            # ydata = signal.sosfilt(sos, ydata)


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
    
    def free_memory(self):
        '''
        Free up memory when done with the analysis
        WARNING: this will break any references to the raw data
        '''
        self.x = None
        self.y = None
        self.sumsignal = None
        self.fg = None

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
        
        longdatax = []
        longdatay = []
        longdatasum = []
        longdatafg = []
        powerspecx = np.zeros(2501)
        powerspecy = np.zeros(2501)

        fhs = sorted(os.listdir(fh), key=file_sort)
        fhs = fhs[:-1] # remove the last file, as this causes errors
        count = 0
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
                    count += 1
                
        
        
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
        initial = 4 # starting frequency for the fit
        list = [i for i in range(len(f)) if f[i] > 4000] 
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
        return Sx, Sy, alphax, alphay, np.mean(xb)/Sx, np.mean(yb)/Sy 


    
    
    def see_calibration(self):
        '''See the calibration of the trap for the dataset'''
        pass
        

    def calibrate_function_gen(self, poly_fit=(1,(-1000, -200)) ,lin_range = [-200*1e-9,150*1e-9], show_plots=['lin_fit']):
        '''Calibrate the function generator
        Args:
        poly_fit: degree of polynomial to fit, default is 1 (linear fit), (range of bead positions to fit)
        lin_range: range of y values to fit the linear region, default is [-200nm, 150nm]
        show_plots: list of plots to show, default has only the 'lin_fit'

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
        jumpy_idxs = absdiff > (np.mean(absdiff)*100)
        
        if plots['signal_jumps']:
            fig = plt.figure(figsize=(10, 5))
            fig.suptitle('Function Generator Signal Jumps')
            ax = fig.add_subplot(212)
            ax.set_title('Highlighted Jumps in Signal')
            ax.plot(t[1:], fg[1:], 'k')
            ax.plot(t[1:][jumpy_idxs], fg[1:][jumpy_idxs], 'r.')
            ax.set_ylabel('Signal (V)')
            plt.xlabel('Time (s)')
            bx = fig.add_subplot(211)
            bx.set_title('Absolute Differences')
            bx.plot(t[1:], absdiff, 'k,')
            bx.set_ylabel('Signal Difference (V)')
            
        
        # Reduce signal jumps to a single point i.e. throw out everything but the end of the jump
        # Also convert from boolean indexing to integer indexing
        idx = []
        previous: bool = jumpy_idxs[0]
        for i, current in enumerate(jumpy_idxs):
            if not current and previous: # Last point was a jump, and this is not a jump
                idx.append(i-1) #Marking the end of jumps for exclusive indexing
            previous = current
        
        # compile the standard deviation of the signal in the stable regions
        stdfg = []
        for i in range(len(idx)-1):
            rangeval = (t[idx[i]] < t) * (t < t[idx[i+1]])
            stdfg.append(np.std(np.abs(fg[rangeval])))
        stdfg = np.array(stdfg)
        
        # get the avg fg and y in each stable region
        avg_fg = []
        avg_y = []
        threshold = 1.1*np.mean(stdfg) # setting the threshold to the mean of the std of the signal
        for i in range(len(stdfg)-1):
            if stdfg[i] < threshold: # ensure the fg is stable (i.e. we're not in a jump)
                rangeval = (t[idx[i] + int(fs/2 * .75)] < t) * (t < t[idx[i] + int(fs/2 * .95)]) #getting the last 0.2 seconds of the signal
                meanfg = np.mean(fg[rangeval])
                meany = np.mean(y[rangeval])
                avg_fg.append(meanfg)
                avg_y.append(meany)
        
        avg_fg = np.array(avg_fg)
        avg_y = np.array(avg_y)

        # Fit the avg fg to the avg y in our linear region
        plt.figure()
        upper_bound = lin_range[1] + self.trapy
        lower_bound = lin_range[0] + self.trapy
        inrange = (lower_bound < avg_y) * (avg_y < upper_bound)
        a, b = np.polyfit(avg_fg[inrange], 1e9*avg_y[inrange], 1)   
        plt.plot(avg_fg, 1e9*avg_y, 'k.')
        plt.plot(avg_fg[inrange], 1e9*avg_y[inrange], 'r.', alpha=0.25)
        plt.plot(avg_fg[inrange], a*avg_fg[inrange] + b, 'g*')
        plt.plot(0, 1e9*self.trapy, 'pb', label='Trap Center') 
        self.a=a
        self.b=b

        if poly_fit[0] > 1:
            bead_pos = a*avg_fg + b
            detector_V = avg_y*self.Sy

            index = (bead_pos > poly_fit[1][0]) * (bead_pos < poly_fit[1][1])
            coeffs = np.polyfit(detector_V[index], bead_pos[index], poly_fit[0])
            print('coeffs:', coeffs)
            
            plt.figure()
            plt.plot(detector_V, bead_pos, 'k.')
            plt.plot(detector_V[index], bead_pos[index], 'gs', alpha=0.25)
            plt.plot(detector_V, np.polyval(coeffs, detector_V), 'r.')
            plt.xlabel('Detector Voltage (V)')
            plt.ylabel('Bead Position (nm)')
            self.yBACKUP = self.y
            y_copy = self.y*self.Sy
            y_copy = np.polyval(coeffs, y_copy)
            self.y=1e-9*y_copy

    def reset_y(self):
        '''Reset the y data to the original data'''
        self.y = self.yBACKUP
        self.yBACKUP = None



                

class Passive_Analysis(OTdataset):
    '''Class for processing passive OT datasets'''
    def __init__(self, fh, params=None):
        '''fh: file handle for where brownian folder can be found, subhandle: relative file handle of the passive dataset i.e. bead1p/'''
        super().__init__(fh, params)
        
        # Load the data
        self.fh = fh +'/'
        self.x, self.y, self.sumsignal, self.fg = self._load_data(self.fh)
        self.x/=self.Sx
        self.y/=self.Sy
        self.t = np.arange(len(self.x))*dt
    
    def final_analysis(self):
        '''Run the final analysis on the passive data, resulting in self.G1, self.G2'''
        x,y,fg,t = self.x, self.y, self.fg, self.t

        avg_msd = np.zeros(int(0.1*fs))
        msd_df:pd.DataFrame = calcmsd2(x, y, t[-1], 20) #max tau of 20 seconds

        alpha_list = []
        omega_list = []
        logtau_list = []
        logmsd_list = []
        tau_list = []
        msd_list = []
        analsis_length = 20 #???

        def linear_alpha(x, m, b):
            return m*x + b
        
        N = 0.2 # iteration steps

        msd = msd_df.msd[:int(analsis_length*fs)]
        tau = msd_df.tau[:int(analsis_length*fs)]
        logmsd = np.log10(msd)
        logtau = np.log10(tau)

        if True:
            plt.figure()
            plt.plot(logtau, logmsd, 'k')
            plt.xlabel('log(tau)')
            plt.ylabel('log(MSD)')
            plt.title('Log-Log plot of MSD vs tau')
            plt.grid()
        
        for i in np.arange(-5, np.log10(5)-N, N):
            idx = (i < logtau) * (logtau < i+N)

            logmsd_list.append(np.mean(logmsd[idx]))
            logtau_list.append(np.mean(logtau[idx]))
            tau_list.append(np.mean(tau[idx]))
            msd_list.append(np.mean(msd[idx]))

            omega_list.append(1/(10**logtau_list[-1]))

            if len(logtau_list) > 1:
                slope = (logmsd_list[-1] - logmsd_list[-2]) / (logtau_list[-1] - logtau_list[-2])
            else:
                slope = float("NaN")
            
            alpha_list.append(slope)
        
        if True:
            plt.figure()
            plt.suptitle('Passive Analysis Parameters')
            plt.subplot(2,1,1)
            plt.plot(logtau_list, logmsd_list, 'ko')
            plt.ylabel('log(msd)')

            plt.subplot(2,1,2)
            plt.plot(logtau_list, alpha_list, 'ko')
            plt.xlabel('log(tau)')
            plt.ylabel('alpha')

        d = 2 # dimensionality of the system
        T = 293.15 # room temperature in K
        K_B = 1.38064852e-23 # Boltzmann constant in J/K
        R = (2e-6)/2 # radius of the bead in m
        
        G1 = []
        G2 = []

        for i in range(1,len(omega_list)):
            gamma = (0.457*(1+alpha_list[i])**2) - (1.36*(1+alpha_list[i])) + 1.9 #approx value from Mason 2000 p.373 below eqn. 8
            Gstar = abs((2*d*K_B*T)/(6*np.pi*R*msd_list[i]*gamma))
            G1temp = Gstar*np.cos(alpha_list[i]*np.pi/2)
            G2temp = Gstar*np.sin(alpha_list[i]*np.pi/2)
            G1.append(G1temp)
            G2.append(G2temp)
        G1 = np.array(G1)
        G2 = np.array(G2)
        omega_list = np.array(omega_list[1:])

        self.G1 = G1
        self.G2 = G2
        self.omegalist = omega_list

        plt.figure()
        plt.loglog(omega_list, G1, 'k*', label='G\'')
        plt.loglog(omega_list, G2, 'r*', label='G\'\'')
        plt.xlabel('Angular Frequency (Radians)', )
        plt.ylabel('Viscoelastic Moduli (Pa)')
        plt.legend()





        
    

class Active_Analysis(OTdataset):
    '''Class for processing active OT datasets'''
    def __init__(self, fh, calibration_params=None):
        '''fh: file handle of the active dataset i.e. bead1 !!!NO TAILING SLASH!!! '''
        super().__init__(fh, calibration_params)

        # Load the data
        self.x, self.y, self.sumsignal, self.fg = self._load_data(self.fh+'/')
        self.x/=self.Sx 
        self.y/=self.Sy 
        self.t = np.arange(len(self.x))*dt
    
    def final_analysis(self, show_freq_plot=True):
        '''Run the final analysis on the active data, resulting in self.G1, self.G2'''
        y,fg,t = self.y, self.fg, self.t #just for ease
        # Calculate the power spectrum of the data
        frequences = []
        for i in range(0,len(fg)-fs,fs): #second chunks
            fgchunk = fg[i:i+fs]
            tchunk = t[i:i+fs]
            Y = scipy.fft.fft(fgchunk)
            N = len(fgchunk)
            n = np.arange(N)
            T = N/fs
            freq = n/T
            index = (freq > 0.1) * (freq < 20000)
            freq = freq[index]
            Y = np.abs(Y[index])
            top_freq = freq[Y == np.max(Y)]
            frequences.append(top_freq[0])
        
        frequences = np.array(frequences)
        changeinfreq = (np.diff(frequences) > 0.5)
        freq_changes = np.arange(len(changeinfreq))[changeinfreq]

            

        print('Frequency changes:', freq_changes)
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
        
        for i in range(len(freq_changes)-1):
            if i == 0:
                f = 1
                start = 0
                end = freq_changes[0]*fs
            else:
                f = frequences[freq_changes[i]]
                start = (freq_changes[i-1]+1)*fs + (fs // 2) #catching tailing ends of frequencies by throwing out starting half second of data
                end = (freq_changes[i])*fs
                
            T = 1/f #period
            y = self.y[start:end]
            fg = self.fg[start:end]
            t = np.arange(len(fg))*dt

            if len(fg) == 0:
                print(f'function generator signal is empty for {start/fs} - {end/fs} seconds segment')
                continue

            fgfit = fg - np.mean(fg) # in V
            OTdata = self.a * fg + self.b # in Nm
            
            # Fit the fg data to a sine
            params, params_cov = scipy.optimize.curve_fit(sinefit, t, fgfit, p0=[np.std(fgfit)*np.sqrt(2), 0, 0, 2*np.pi*f])
            if np.abs(np.std(fgfit)*np.sqrt(2)/params[0]) > 2:
                params, params_cov = scipy.optimize.curve_fit(sinefit, t, fgfit, p0=[np.std(fgfit)*np.sqrt(2), np.pi, 0, 2*np.pi*f])
            phase = params[1]
            fg_freq = params[3]/(2*np.pi)
            freqlist.append(fg_freq) 

            if show_freq_plot:
                plt.figure(tight_layout=True)
                plt.subplot(311)
                plt.plot(t, fg, 'k')
                plt.plot(t, sinefit(t, *params), 'r-')
                plt.title(f'{i}th Frequency: {fg_freq} Hz')

            Fcurve = -(self.ky / 1e-12 * 1e-9) * (((1e9*y-np.mean(1e9*y)) - (OTdata-np.mean(OTdata)))) # pN

            # Fit the y data to a sine
            Noscil = 5
            oscil_freq = fg_freq
            params, params_cov = scipy.optimize.curve_fit(sinefit2(2*np.pi*oscil_freq), t, y, p0=[np.std(y)*np.sqrt(2), 0, 0])
            params = np.append(params, 2*np.pi*oscil_freq)
            phase = params[1]

            if show_freq_plot: 
                plt.subplot(312).set_title('Y Data Averaging (black is average)')

            avg_y = np.zeros(int(Noscil/oscil_freq/dt))
            times = dt*np.arange(len(avg_y))
            count = 0
            for j in range(0, len(y), avg_y.shape[0]):
                if len(y[j:j+int(Noscil/oscil_freq/dt)]) == len(avg_y):
                    count += 1
                    avg_y += y[j:j+int(Noscil/oscil_freq/dt)]
                    if show_freq_plot:
                        plt.plot(times, y[j:j+int((Noscil/oscil_freq)/dt)], ',')
            avg_y /= count
            
            if show_freq_plot:
                plt.plot(times, avg_y, 'k-')
            params, params_cov = scipy.optimize.curve_fit(sinefit2(2*np.pi*oscil_freq), times, avg_y, p0=[np.std(y)*np.sqrt(2), 0, 0])
            params = np.append(params, 2*np.pi*oscil_freq)
            yphase = params[1]
            ynot.append(params[0]) 

            # Fit the forces
            params, params_covariance = scipy.optimize.curve_fit(sinefit2(2*np.pi*oscil_freq), t, Fcurve, p0=[np.std(Fcurve)*np.sqrt(2), 0, 0])
            params = np.append(params, 2*np.pi*oscil_freq)

            if show_freq_plot:
                plt.subplot(313).set_title('Force Averaging (black is average)')
            
            avg_F = np.zeros(int(Noscil/oscil_freq/dt))
            times = dt*np.arange(len(avg_F))
            count = 0
            for j in range(0, len(Fcurve), avg_F.shape[0]):
                if len(Fcurve[j:j+int(Noscil/oscil_freq/dt)]) == len(avg_F):
                    count += 1
                    avg_F += Fcurve[j:j+int(Noscil/oscil_freq/dt)]
                    if show_freq_plot:
                        plt.plot(times, Fcurve[j:j+int((Noscil/oscil_freq)/dt)], ',')
            avg_F /= count
            
            if show_freq_plot:
                plt.plot(times, avg_F, 'k-')
            params, params_cov = scipy.optimize.curve_fit(sinefit2(2*np.pi*oscil_freq), times, avg_F, p0=[np.std(Fcurve)*np.sqrt(2), 0, 0])
            params = np.append(params, 2*np.pi*oscil_freq)

            deltatemp = yphase - params[1]
            if deltatemp < 0:
                deltatemp += 2*np.pi
            delta.append(deltatemp)
            fnot.append(params[0]*1e-12)


        freqlist = np.array(freqlist)
        ynot = np.array(ynot)
        fnot = np.array(fnot)
        delta = np.array(delta)

        # Calculate the G'(G1) and G''(G2)
        G1 = np.abs(fnot/(6*np.pi*1e-6*ynot))*np.cos(delta)
        G2 = np.abs(fnot/(6*np.pi*1e-6*ynot))*np.sin(delta)
        self.G1 = G1
        self.G2 = G2
        self.freqlist = freqlist
        plt.figure()
        plt.loglog(freqlist, G1, 'k*', label='G\'')
        plt.loglog(freqlist, G2, 'r*', label='G\'\'')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Viscoelastic Moduli (Pa)')
        plt.legend()



