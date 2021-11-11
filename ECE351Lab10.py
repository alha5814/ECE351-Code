"""
name: Abdalrahman Alhajri
date: nov 10, 2021
Lab: 10
ECE 351-52

"""


import numpy as np
import scipy.signal as sig
import time
import matplotlib.pyplot as plt



steps = 0.5
w = np.arange(1000, 1000000+steps, steps)
R = 1000
C = 100e-9
L = 27e-3
def h(w, R, C, L):
    
    h_mag = (w/R/C)/(np.sqrt((1/L/C-w**2)**2+(w/R/C)**2))
    h_phase = 90-np.arctan((w/(R*C))/(1/(L*C)-(w**2)))*180/(np.pi)
    for i in range(len(h_phase)):
        if (h_phase[i] > 90):
            h_phase[i] = h_phase[i] - 180
    return h_mag, h_phase


h_mag, h_phase = h(w, R, C, L)


        
plt.figure(figsize = (7, 5))
plt.subplot(2, 1, 1)
plt.semilogx(w, 20*np.log10(h_mag))
plt.ylabel('H_mag (dB)')
plt.title('Figure 1: H(jw)')
plt.grid()
plt.subplot(2, 1, 2)
plt.semilogx(w, h_phase)
plt.ylabel('H_phase (deg)')
plt.xlabel('w(rad/s)')
plt.grid()



num = [0, 1/R/C, 0]
den = [1, 1/R/C, 1/L/C]
W, mag, phase = sig.bode((num,den), w)
plt.figure(figsize = (7, 5))
plt.subplot(2, 1, 1)
plt.semilogx(w, mag)
plt.ylabel('H_mag (dB)')
plt.title('Figure 2: H(jw) built-in')
plt.grid()
plt.subplot(2, 1, 2)
plt.semilogx(w, phase)
plt.ylabel('H_phase (deg)')
plt.xlabel('w(rad/s)')
plt.grid()


f = w/(2*np.pi)
plt.figure(figsize = (7, 7))

import control as con 
sys = con.TransferFunction(num, den)
_ = con.bode(sys, w, dB = True, Hz = True, deg = True, Plot = True)
plt.title('Figure 3: H(jw) built-in')
#%% Part 2

fs = 50000
steps = 1/(fs)
t =np.arange(0, 0.01+steps, steps)

x = np.cos(2*np.pi*100*t) + np.cos(2*np.pi*3024*t) + np.sin(2*np.pi*50000*t)
#x = np.cos(2*np.pi*3024*t)
plt.figure(figsize = (7, 5))
plt.subplot(2, 1, 1)
plt.plot(t, x)
plt.ylabel('x(t)')
plt.title('Figure 4: input signal')
plt.grid()
plt.xlabel('t(s)')

num = [0, 1/R/C, 0]
den = [1, 1/R/C, 1/L/C]

znum, zden = sig.bilinear(num, den, fs*10)

y = sig.lfilter(znum, zden, x)

plt.figure(figsize = (7, 5))
plt.subplot(2, 1, 1)
plt.plot(t, y)
plt.ylabel('y(t)')
plt.title('Figure 5: output signal')
plt.grid()
plt.xlabel('t(s)')










