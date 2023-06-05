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

K = 3
T = 4

start = 0
stop = 30
increment = 0.1
t = np.arange(start,stop,increment)

y = K*(1-np.exp(-t/T))

plt.plot(t,y)
plt.title('1st Order Dynamic System')
plt.xlabel('t [s]')
plt.ylabel('y(t)')
plt.grid()
plt.show()