
"""

name: Abdalrahman Alhajri
date: Sep 23, 2021
Lab: 4
ECE 351-52
comments:
"""
import numpy as np
import scipy.signal as sig
import time
import matplotlib.pyplot as plt


steps = 1e-2 # Define step size

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

t=np.arange(-10 , 10 + steps , steps)

#%% task 1

def h1(t):
    return np.exp(-2*t)*(u(t)-u(t-3))

def h2(t):
    return u(t-2) - u(t-6) 

def h3(t):
    f = 0.25
    w = 2*f*np.pi
    y = np.cos(w*t)*u(t)
    return y


#%% task 2

plt.figure(figsize = (5 ,5))
plt.subplot(3 , 1 , 1)
plt.plot(t , h1(t))
plt.grid()
plt.ylabel('h1(t)')
plt.title('figure 1: Transfer Functions')
plt.subplot(3 , 1 , 2)
plt.plot(t , h2(t))
plt.grid()
plt.ylabel('h2(t)')
plt.subplot(3 , 1 , 3)
plt.plot(t , h3(t))
plt.grid()
plt.ylabel('h3(t)')


#%% task 3

def conv(f1,f2): 
    y= len(f1)
    x= len(f2)
    
    f1Extended = np.append(f1, np.zeros((1, x -1)))
    f2Extended = np.append(f2, np.zeros((1, y -1)))
    
    result = np.zeros(f1Extended.shape)
    
    for i in range(y + x - 2):
        result[i] = 0
        
        for j in range(y):
            if( (i - j + 1) > 0 ):
                result[i] = result[i] + f1Extended[j] * f2Extended[i - j + 1]

    return result



NN=len(t)
tExtended=np.arange(2*t[0], 2*t[NN-1] + steps, steps)


conv12 = conv(h1(t), u(t))*steps

plt.figure(figsize = (5, 5))
plt.subplot(3, 1, 1)
plt.plot(tExtended , conv12, label= 'user-defined')
plt.ylabel('h1(t) * u(t)')
plt.title('figure 2: Step Response (by coding)')
plt.grid()



conv23 = conv(h2(t), u(t))*steps

plt.subplot(3, 1, 2)
plt.plot(tExtended , conv23, label= 'user-defined')
plt.ylabel('h2(t) * u(t)')
plt.grid()



conv13 = conv(h3(t), u(t))*steps

plt.subplot(3, 1, 3)
plt.plot(tExtended , conv13, label= 'user-defined')
plt.ylabel('h3(t) * u(t)')
plt.grid()

#%% task 4

t=np.arange(2*t[0], 2*t[NN-1] + steps, steps)

def f1(t):
    y = (1/2)*(1 - np.exp(-2*t))*u(t) + (1/2)*(np.exp(-2*t) - np.exp(-6))*u(t-3)
    return y

def f2(t):
    return (t-2)*u(t-2) - (t-6)*u(t-6) 

def f3(t):
    f = 0.25
    w = 2*f*np.pi
    y = (1/w)*np.sin(w*t)*u(t)
    return y

plt.figure(figsize = (5 ,5))
plt.subplot(3 , 1 , 1)
plt.plot(t , f1(t))
plt.grid()
plt.ylabel('h1(t) * u(t)')
plt.title('figure 3: Step Response (by hand)')
plt.subplot(3 , 1 , 2)
plt.plot(t , f2(t))
plt.grid()
plt.ylabel('h2(t) * u(t)')
plt.subplot(3 , 1 , 3)
plt.plot(t , f3(t))
plt.grid()
plt.ylabel('h3(t) * u(t)')