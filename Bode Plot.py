import numpy as np
import matplotlib.pyplot as plt
import control

import warnings
warnings.filterwarnings('ignore')

# requires coefficients of the numerator and denominator polynomials
# the coefficients are given starting with the highest power of s

G = 10*control.tf([1,1],[0.1,1])
print(G)

w = np.logspace(-1.5,1,200)  # 200 frequencies between 0.03 and 10 rad/s
mag,phase,omega = control.bode(G,w,Hz=True,deg=True,Plot=True)
plt.show()

