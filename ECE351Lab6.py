"""
name: Abdalrahman Alhajri
date: oct 7, 2021
Lab: 6
ECE 351-52
"""
import numpy as np
import scipy.signal as sig
import time
import matplotlib.pyplot as plt
import sympy 
'''
s = sympy.symbols('s')
num = s**2+6*s+12
den = s**3+10*s**2+24*s
y = num / den
partial = sympy.apart(y)

print(partial)
'''
steps= 1e-3 # Define step size

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
        if t[i] >= 0:
            y[i] = 1
        else:
            y[i] = 0
            
    return y

t=np.arange(0 , 2 + steps , steps)

#%% task 1

def y(t):
    return (np.exp(-6*t)-0.5*np.exp(-4*t)+0.5)*u(t)


plt.figure(figsize = (5 ,5))
plt.subplot(2, 1, 1)
plt.plot(t , y(t))
plt.grid()
plt.xlim(0, 2)
plt.ylabel('user-defined')
plt.title('figure 1: step response')
#%% task 2

num = [1, 6, 12]
den = [1, 10, 24]

tout , yout = sig.step((num, den), T = t)

plt.subplot(2, 1, 2)
plt.plot(tout , yout)
plt.grid()
plt.xlim(0, 2)
plt.ylabel('built-in')

#%%task 3


num1 = [1, 6, 12]
den1 = [1, 10, 24, 0]

R1, P1, K1 = sig.residue(num1, den1, steps, rtype='avg')

print('residue: ', R1)
print('poles: ', P1)
print('gain: ', K1)

#%% task 4

num2 = [25250]
den2 = [1, 18, 218, 2036, 9085, 25250, 0]

R2, P2, K2 = sig.residue(num2, den2, steps, rtype='avg')

print('\nR: ', R2)
print('P: ', P2)
print('K: ', K2)

#%% task 5

t=np.arange(0 , 4.5 + steps , steps)

def cosine_method(R,P,t):
    y = 0
    for i in range(len(R)):
        R_mag = np.abs(R[i])
        R_ang = np.angle(R[i])
        alpha = np.real(P[i])
        omega = np.imag(P[i])
        y = y + R_mag*np.exp(alpha*t)*np.cos(omega*t+R_ang)*u(t)
    
    return y

plt.figure(figsize = (5 ,5))
plt.subplot(2, 1, 1)
plt.plot(t , cosine_method(R2,P2,t))
plt.grid()
plt.ylabel('user-defined')
plt.title('figure 2: step response')

#%% task 6

num3 = [25250]
den3 = [1, 18, 218, 2036, 9085, 25250]

tout , yout = sig.step((num3, den3), T = t)

plt.subplot(2, 1, 2)
plt.plot(tout , yout)
plt.grid()
plt.ylabel('built-in')
