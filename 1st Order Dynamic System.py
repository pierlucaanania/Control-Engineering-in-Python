'''
System described by:
--------------------
y_dot = ay+bu
or
y_dot = 1/T(-y+Ku)
--------------------
where: K is the gain and T is time constant, u(t)=U is a step.
Solution:

y(t) = KU(1-exp(-t/T))
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

K = 3
T = 4
U = 5

start = 0
stop = 30
increment = 0.1
t = np.arange(start,stop,increment)

y = K*U*(1-np.exp(-t/T))

plt.plot(t,y)
plt.title('1st Order Dynamic System')
plt.xlabel('t [s]')
plt.ylabel('y(t)')
plt.grid()
plt.show()

### ODE Solver

tstart = 0
tstop = 25
increment = 1
y0 = 1
t = np.arange(tstart,tstop+1,increment)

#Function that return y_dot or dy/dt

def system1order(y, t, K, T, U):
    dydt = (1/T)*(-y + K*U)
    return dydt

x = odeint(system1order, y0, t, args=(K,T,U))

plt.plot(t,x)
plt.title('1st Order System ODE')
plt.xlabel('t [s]')
plt.ylabel('y(t)')
plt.grid()
plt.show()