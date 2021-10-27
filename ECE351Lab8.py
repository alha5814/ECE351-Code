"""
name: Abdalrahman Alhajri
date: oct 21, 2021
Lab: 8
ECE 351-52

"""

import numpy as np
import scipy.signal as sig
import time
import matplotlib.pyplot as plt


a = np.zeros((4, 1))
b = np.zeros((4, 1))

for k in np.arange(1, 4):
    b[k] = 2/(k*np.pi) * (1-np.cos(k*np.pi))
    
for k in np.arange(1, 4):
    a[k] = 0

a0 = a[0]
a1 = a[1]
b1 = b[1]
b2 = b[2]
b3 = b[3]

print('a_0 = ', a0)
print('a_1 = ', a1)
print('b_1 = ', b1)
print('b_2 = ', b2)
print('b_3 = ', b3)

steps = 1e-3
t=np.arange(0 , 20 + steps , steps)
T = 8 
y = 0

N = [1, 3, 15, 50, 150, 1500]


for h in [1, 2]:
    for i in ([1+(h-1)*3, 2+(h-1)*3, 3+(h-1)*3]):
        for k in np.arange(1, N[i-1]+1):
            
            b = 2/(k*np.pi) * (1-np.cos(k*np.pi))
            
            x = b*np.sin(k*2*np.pi*t/T)
            
            y = y + x
            
        plt.figure(h, figsize = (5 ,5))
        plt.subplot(3, 1, i-(h-1)*3)
        plt.plot(t , y)
        plt.grid()
        plt.ylabel('N = %i' % N[i-1])
        if i == 1 or i == 4:
            plt.title('Fourier Series Approximations of x(t)')
        if i == 3 or i == 6:
            plt.xlabel('t [s]')
            plt.show()
        y = 0
