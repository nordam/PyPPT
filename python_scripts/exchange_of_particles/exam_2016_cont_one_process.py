# example to run:
"""
mpiexec -n 1 python exam_2016_cont_one_process.py
"""
# where 1 is the number of processes

import numpy as np

from mpi4py import MPI
from matplotlib import pyplot as plt
plt.style.use('bmh')

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
mpi_size = comm.Get_size()

### INITIALIZING global parameters/properties

# time and parameters of current veolicity field
t = 0.0
A = 0.1
e = 0.25
w = 1

# transport parameters
tmax = 4.0
dt   = 0.1

# grid of points
# specify resolution
Nx = 50
Ny = 25
# specify x- and y limits,
x_min = 0
y_min = 0
x_max = 2 # x_max
y_max = 1 # y_max
x_plot_margin = x_max/10
y_plot_margin = y_max/10

### INITIALIZING functions

# implementation of Eq. (1) in the exam set
def doublegyre(x, y, t, A, e, w):
    a = e * np.sin(w*t)
    b = 1 - 2*e*np.sin(w*t)
    f = a*x**2 + b*x
    return np.array([
            -np.pi*A*np.sin(np.pi*f) * np.cos(np.pi*y),              # x component of velocity
             np.pi*A*np.cos(np.pi*f) * np.sin(np.pi*y) * (2*a*x + b) # y component of velocity
        ])

# wrapper function to pass to integrator
# X0 is a two-component vector [x, y]
def f(X, t):
    # Parameters of the velocity field
    A = 0.1
    e = 0.25 # epsilon
    w = 1    # omega
    return doublegyre(X[0], X[1], t, A, e, w)

# 4th order Runge-Kutta integrator
# X0 is a two-component vector [x, y]
def rk4(X, t, dt, f):
    k1 = f(X,           t)
    k2 = f(X + k1*dt/2, t + dt/2)
    k3 = f(X + k2*dt/2, t + dt/2)
    k4 = f(X + k3*dt,   t + dt)
    return X + dt*(k1 + 2*k2 + 2*k3 + k4) / 6
	
# function to calculate a trajectory from an
# initial position X0 at t = 0, moving forward
# until t = tmax, using the given timestep and
# integrator
def trajectory(X0, tmax, dt, integrator, f):
    t    = 0
    # Number of timesteps
    Nx = int(tmax / dt)
    # Array to hold the entire trajectory
    PX = np.zeros((2, Nx+1))
    # Initial position
    PX[:,0] = X0
    # Loop over all timesteps
    for i in range(1, Nx+1):
        PX[:,i] = integrator(PX[:,i-1], t, dt, f)
        t += dt
    # Return entire trajectory
    return PX

### INITIALIZING end

print("total number of ranks:", mpi_size, "\nrank:", rank, "\n")

### velocity vector field

x = np.linspace(x_min, x_max, Nx)
y = np.linspace(y_min, y_max, Ny)
X, Y = np.meshgrid(x, y)

# Calculate field
V = doublegyre(X, Y, t, A, e, w)

# Vector plot
fig = plt.figure(figsize = (16,8))
plt.quiver(x, y, V[0], V[1], linewidths = -1.0, scale = 10, alpha = 0.6)
# add text showing time, and set plot limits
plt.text(1.65, 0.9, '$t = %s$' % t, size = 36)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.show()
#plt.draw()

### Plot trajectories

# Using Runge-Kutta

X0   = [1.12, 0.6]
tmax = 50

# Set figure size
fig = plt.figure(figsize = (12,6))

# Plot trajectory for dt = 0.1
dt   = 0.1
T = trajectory(X0, tmax, dt, rk4, f)
plt.plot(T[0,:], T[1,:], label = 'dt = %s' % dt)

# Plot trajectory for dt = 0.01
dt   = 0.01
T = trajectory(X0, tmax, dt, rk4, f)
plt.plot(T[0,:], T[1,:], '--', label = 'dt = %s' % dt)

# Add legend and set limits
plt.legend()
plt.xlim(0,2)
plt.ylim(0,1)
plt.show()