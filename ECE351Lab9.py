"""
name: Abdalrahman Alhajri
date: oct 28, 2021
Lab: 9
ECE 351-52

"""


import numpy as np
import scipy.signal as sig
import time
import matplotlib.pyplot as plt
import scipy.fftpack 


fs = 100

T = 1/fs
t = np.arange(0, 2, T)

x = np.cos(2*np.pi*t)

    
def ffs(x, fs):
    N = len(x) 
    
    X_fft = scipy.fftpack.fft(x) 
    X_fft_shifted = scipy.fftpack.fftshift(X_fft) 
    
    freq = np.arange(-N/2, N/2) * fs/N 

    X_mag = np.abs(X_fft_shifted)/N
    X_phi = np.angle(X_fft_shifted)
    for i in range(len(X_phi)):
        if np.abs(X_mag[i]) < 1e-10:
            X_phi[i] = 0

    return freq, X_mag, X_phi
        

plt.figure(figsize = (7, 5))
plt.subplot(3, 1, 1)
plt.plot(t, x)
plt.ylabel('x(t)')
plt.title('figure 1:')
plt.grid()
plt.subplot(3, 2, 3)
plt.ylabel('Mag')
plt.grid()
plt.subplot(3, 2, 4)
plt.xlim(-2, 2)
plt.grid()
plt.subplot(3, 2, 5)
plt.ylabel('Phase')
plt.grid()
plt.xlabel('freq(Hz)')
plt.subplot(3, 2, 6)
plt.xlim(-2, 2)
plt.grid()
plt.xlabel('freq(Hz)')
plt.show()

x = 5*np.sin(2*np.pi*t)
freq, X_mag, X_phi = ffs(x, fs)

plt.figure(figsize = (7, 5))
plt.subplot(3, 1, 1)
plt.plot(t, x)
plt.ylabel('x(t)')
plt.xlabel('t(s)')
plt.title('figure 2:')
plt.grid()
plt.subplot(3, 2, 3)
plt.stem(freq, X_mag)
plt.ylabel('Mag')
plt.grid()
plt.subplot(3, 2, 4)
plt.stem(freq, X_mag)
plt.xlim(-2, 2)
plt.grid()
plt.subplot(3, 2, 5)
plt.stem(freq, X_phi) 
plt.ylabel('Phase')
plt.grid()
plt.xlabel('freq(Hz)')
plt.subplot(3, 2, 6)
plt.stem(freq, X_phi)
plt.xlim(-2, 2)
plt.xlabel('freq(Hz)')
plt.grid()
plt.show()



x = 2*np.cos((2*np.pi*2*t)-2) + (np.sin((2*np.pi*6*t)+3))**3
freq, X_mag, X_phi = ffs(x, fs)

plt.figure(figsize = (7, 5))
plt.subplot(3, 1, 1)
plt.plot(t, x)
plt.ylabel('x(t)')
plt.xlabel('t(s)')
plt.title('figure 3:')
plt.grid()
plt.subplot(3, 2, 3)
plt.stem(freq, X_mag)
plt.ylabel('Mag')
plt.grid()
plt.subplot(3, 2, 4)
plt.stem(freq, X_mag)
plt.xlim(-15, 15)
plt.grid()
plt.subplot(3, 2, 5)
plt.stem(freq, X_phi) 
plt.ylabel('Phase')
plt.grid()
plt.xlabel('freq(Hz)')
plt.subplot(3, 2, 6)
plt.stem(freq, X_phi)
plt.xlim(-15, 15)
plt.xlabel('freq(Hz)')
plt.grid()
plt.show()

T = 8
steps = 1e-2
t = np.arange(0, 16+steps, steps)
y = 0
for k in np.arange(1, 16):
    b = 2/(k*np.pi) * (1-np.cos(k*np.pi))
    x = b*np.sin(k*2*np.pi*t/T)
    y = y + x

freq, X_mag, X_phi = ffs(y, fs)

plt.figure(figsize = (7, 5))
plt.subplot(3, 1, 1)
plt.plot(t, y)
plt.ylabel('x(t)')
plt.xlabel('t(s)')
plt.title('figure 4:')
plt.grid()
plt.subplot(3, 2, 3)
plt.stem(freq, X_mag)
plt.ylabel('Mag')
plt.grid()
plt.subplot(3, 2, 4)
plt.stem(freq, X_mag)
plt.xlim(-2, 2)
plt.grid()
plt.subplot(3, 2, 5)
plt.stem(freq, X_phi) 
plt.ylabel('Phase')
plt.grid()
plt.xlabel('freq(Hz)')
plt.subplot(3, 2, 6)
plt.stem(freq, X_phi)
plt.xlim(-2, 2)
plt.xlabel('freq(Hz)')
plt.grid()
plt.show()



