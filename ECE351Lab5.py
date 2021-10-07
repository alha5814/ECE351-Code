#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 11:37:14 2021

@author: abdulrahmanalhajre
"""

"""
name: Abdalrahman Alhajri
date: Sep 30, 2021
Lab: 5
ECE 351-52

"""
import numpy as np
import scipy.signal as sig
import time
import matplotlib.pyplot as plt

steps= 1e-7 # Define step size

def r(t):
    y=np.zeros((len(t),1))
    for i in range(len(t)):
        if t[i] <= 0:
            y[i] = 0
        else:
            y[i] = t[i]
    return y

def u(t):
    y=np.zeros(t.shape)
    for i in range(len(t)):
        if t[i] <= 0:
            y[i] = 0
        else:
            y[i] = 1
    return y

t=np.arange(0 , 1.2e-3 + steps , steps)

#%% task 1

R = 1000
L = 27e-3 
C = 100e-9

def sine_method(R,L,C,t):
    alpha = -1/(2*R*C)
    w = (1/2)*np.sqrt((1/(R*C))**2-4*(1/(np.sqrt(L*C)))**2+0*1j)
    p = alpha + w
    g = 1/(R*C)*p
    g_mag = np.abs(g)
    g_rad = np.angle(g)
    g_deg = g_rad * 180/(np.pi) 
    
    y = (g_mag/np.abs(w))*np.exp(alpha*t)*np.sin(np.abs(w)*t + g_rad)*u(t)
    
    return y

plt.figure(figsize = (5 ,5))
plt.subplot(2, 1, 1)
plt.plot(t , sine_method(R,L,C,t))
plt.grid()
plt.ylabel('user-defined')
plt.title('figure 1: impulse response')

#%% task 2 

num = [0, 10000, 0]
den = [1, 10000, 3.7037e8]

tout , yout = sig.impulse((num, den), T = t)

plt.subplot(2, 1, 2)
plt.plot(tout , yout)
plt.grid()
plt.ylabel('built-in')

#%% task 3

num = [0, 1/(R*C), 0]
den = [1, 1/(R*C), 1/(C*L)]

tout , yout = sig.step((num, den), T = t)

plt.figure(figsize = (5 ,5))
plt.subplot(2, 1, 1)
plt.plot(tout , yout)
plt.grid()
plt.ylabel('built-in')
plt.title('figure 2: step response')




