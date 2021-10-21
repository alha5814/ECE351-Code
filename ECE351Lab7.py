"""
name: Abdalrahman Alhajri
date: oct 14, 2021
Lab: 7
ECE 351-52

"""

import numpy as np
import scipy.signal as sig
import time
import matplotlib.pyplot as plt
#%% part 1

numG = [1, 9]
denG = [1, -2, -40, -64]

numA = [1, 4]
denA = [1, 4, 3]

B = [1, 26, 168]

ZG, PG, KG = sig.tf2zpk(numG, denG)
print('for G(s):')
print('Zeros:', ZG)
print('Poles:', PG)
print('Gain:', KG)

ZA, PA, KA = sig.tf2zpk(numA, denA)
print('\nfor A(s):')
print('Zeros:', ZA)
print('Poles:', PA)
print('Gain:', KA)

ZB = np.roots(B)
print('\nfor B(s):')
print('Zeros:', ZB)

numO = sig.convolve(numG, numA)
denO = sig.convolve(denG, denA)
print('\nfor H_o(s):')
print('Num:', numO)
print('Den:', denO)

steps = 1e-3
t=np.arange(0 , 2 + steps , steps)
tout , yout = sig.step((numO, denO), T = t)

plt.figure(figsize = (5 ,5))
plt.subplot(2, 1, 1)
plt.plot(tout , yout)
plt.grid()
plt.ylabel('open loop')
plt.title('figure 1: step response for the system')

#%% part 2

numC = sig.convolve(numA, numG)
denC1 = sig.convolve(numG, B)
denC2 = sig.convolve(denA, denC1)
denC3 = sig.convolve(denA, denG)
denC4 = np.array(denC2) + np.array(denC3)

ZC, PC, KC = sig.tf2zpk(numC, denC4)
print('\nfor Close loop:')
print('Zeros:', ZC)
print('Poles:', PC)
print('Gain:', KC)
t=np.arange(0 , 4 + steps , steps)
tout , yout = sig.step((numC, denC4), T = t)

plt.figure(figsize = (5 ,5))
plt.subplot(2, 1, 1)
plt.plot(tout , yout)
plt.grid()
plt.ylabel('closed loop')
plt.title('figure 2: step response for the system')

