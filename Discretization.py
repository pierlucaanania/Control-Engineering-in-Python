import numpy as np
import matplotlib.pyplot as plt

#Model parameters
K = 3
T = 4
a = -1/T
b = K/T

#Simulation parameters
Ts = 0.1
Tstop = 30
uk = 1                      #step response
yk = 0                      #initial value
N = int(Tstop/Ts)           #simulation length
data = []
data.append(yk)

#Simulation
for k in range(N):
    yk1 = (1+a*Ts)*yk + b*Ts*uk
    yk = yk1
    data.append(yk1)

#Plotting
t = np.arange(0,Tstop+Ts,Ts)
plt.plot(t,data)
plt.xlabel('Time [s]')
plt.ylabel('y(t)')
plt.ylim([0,3.5])
plt.title('Discretization of a first order system')
plt.grid()
plt.show()

