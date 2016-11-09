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

## time and parameters of current veolicity field
    
t_0 = 0.0
t_plot = 2
A = 0.1
e = 0.25 # epsilon
w = 1 # omega

## transport parameters
t_max = 10
# point of interest
XY_1   = [1.12, 0.6]
XY_2   = [1.14, 0.6]
XY_3   = [1.13, 0.58]
XY_4   = [1.13, 0.62]
# time step in Runge-Kutta integrator
dt   = 0.1
# multiplication factor to compare different time steps 
dt_factor = 10

## grid of points
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

# Specify limit of particle rectangle
particles_x_min = 0.9
particles_x_max = 1.1
particles_y_min = 0.4
particles_y_max = 0.6

# fig size here?

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
# XY is a two-component vector [x, y]
def f(X, t):
    # Parameters of the velocity field is given at the start of this file
    return doublegyre(X[0], X[1], t, A, e, w)

# 4th order Runge-Kutta integrator
# XY is a two-component vector [x, y]
def rk4(X, t, dt, f):
    k1 = f(X,           t)
    k2 = f(X + k1*dt/2, t + dt/2)
    k3 = f(X + k2*dt/2, t + dt/2)
    k4 = f(X + k3*dt,   t + dt)
    return X + dt*(k1 + 2*k2 + 2*k3 + k4) / 6
	
# function to calculate a trajectory from an
# initial position XY at t = 0, moving forward
# until t = t_max, using the given timestep and
# integrator
def trajectory(XY, t_max, dt, integrator, f):
    t = t_0
    # Number of timesteps
    Nx = int(t_max / dt)
    # Array to hold the entire trajectory
    PX = np.zeros((2, Nx+1))
    # Initial position
    PX[:,0] = XY
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
V = doublegyre(X, Y, t_plot, A, e, w)

# Vector plot
fig = plt.figure(figsize = (16,8))
plt.quiver(x, y, V[0], V[1], linewidths = -1.0, scale = 10, alpha = 0.6)
# add text showing time, and set plot limits
plt.text(1.65, 0.9, '$t = %s$' % t_plot, size = 36)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.show()
#plt.draw()

### plot trajectories

# Using Runge-Kutta

# Set figure size
fig = plt.figure(figsize = (12,6))

# Plot trajectory for dt (dt initialized at the start of this file)

#T = trajectory(XY_1, t_max, dt, rk4, f)
#plt.plot(T[0,:], T[1,:], label = 'dt = %s' % dt)

# Plot trajectory for dt = dt*dt_factor
#T = trajectory(XY, t_max, dt*dt_factor, rk4, f)
#plt.plot(T[0,:], T[1,:], '--', label = 'dt = %s' % (dt*dt_factor))

# Plot trajectory for 4 points:
T = trajectory(XY_1, t_max, dt, rk4, f)
plt.plot(T[0,:], T[1,:], label = 'x,y = %s' % XY_1)

T = trajectory(XY_2, t_max, dt, rk4, f)
plt.plot(T[0,:], T[1,:], label = 'x,y = %s' % XY_2)

T = trajectory(XY_4, t_max, dt, rk4, f)
plt.plot(T[0,:], T[1,:], label = 'x,y = %s' % XY_3)

T = trajectory(XY_4, t_max, dt, rk4, f)
plt.plot(T[0,:], T[1,:], label = 'x,y = %s' % XY_4)

# Add legend and set limits
plt.legend()
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.show()

### plot grid of points before and after transport

# Create lists of all x and y positions
X  = np.linspace(x_min, x_max, Nx)
Y  = np.linspace(y_min, y_max, Ny)

# Array to hold all grid points
XY = np.zeros((2, Ny, Nx))
# Use meshgrid to turn lists into rank 2 arrays
# of x and y positions
XY[:] = np.meshgrid(X, Y)

# Make scatter plot to show all grid points
fig = plt.figure(figsize = (12,6))
plt.scatter(XY[0,:], XY[1,:], marker = '.', c = '#348ABD')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.show()

# Array to hold all grid points after transport
X1 = np.zeros((2, Ny, Nx))

# Loop over grid and update all positions
# This is where parallelisation would happen, since
# each position is independent of all the others
for i in range(Nx):
    for j in range(Ny):
        # Keep only the last position, not the entire trajectory
        X1[:,j,i] = trajectory(XY[:,j,i], t_max, dt, rk4, f)[:,-1]
 
# Make scatter plot to show all grid points
fig = plt.figure(figsize = (12,6))
plt.scatter(X1[0,:], X1[1,:], marker = '.', c = '#348ABD')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.show()

### plot particle rectangle before and after transport

# Create lists of all x and y positions
X  = np.linspace(particles_x_min, particles_x_max, Nx)
Y  = np.linspace(particles_y_min, particles_y_max, Ny)

# Array to hold all grid points
XY = np.zeros((2, Ny, Nx))
# Use meshgrid to turn lists into rank 2 arrays
# of x and y positions
XY[:] = np.meshgrid(X, Y)

# Make scatter plot to show all grid points
fig = plt.figure(figsize = (12,6))
plt.scatter(XY[0,:], XY[1,:], marker = '.', c = '#348ABD')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

# transportation

# Array to hold all grid points after transport
X1 = np.zeros((2, Ny, Nx))

# Transport parameters
t_max = 2.0

# Loop over grid and update all positions
# This is where parallelisation would happen, since
# each position is independent of all the others
for i in range(Nx):
    for j in range(Ny):
        # Keep only the last position, not the entire trajectory
        X1[:,j,i] = trajectory(XY[:,j,i], t_max, dt, rk4, f)[:,-1]

        
# Make scatter plot to show all grid points
fig = plt.figure(figsize = (12,6))
plt.scatter(X1[0,:], X1[1,:], marker = '.', c = '#348ABD')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.show()
savefig()

# Array to hold all grid points after transport
X1 = np.zeros((2, Ny, Nx))

# Transport parameters
t_max = 8.0

# Loop over grid and update all positions
# This is where parallelisation would happen, since
# each position is independent of all the others
for i in range(Nx):
    for j in range(Ny):
        # Keep only the last position, not the entire trajectory
        X1[:,j,i] = trajectory(XY[:,j,i], t_max, dt, rk4, f)[:,-1]

        
# Make scatter plot to show all grid points
fig = plt.figure(figsize = (12,6))
plt.scatter(X1[0,:], X1[1,:], marker = '.', c = '#348ABD')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.show()
savefig()

# # Array to hold all grid points after transport
# X1 = np.zeros((2, Ny, Nx))

# # Transport parameters
# t_max = 16.0

# # Loop over grid and update all positions
# # This is where parallelisation would happen, since
# # each position is independent of all the others
# for i in range(Nx):
    # for j in range(Ny):
        # # Keep only the last position, not the entire trajectory
        # X1[:,j,i] = trajectory(XY[:,j,i], t_max, dt, rk4, f)[:,-1]

        
# # Make scatter plot to show all grid points
# fig = plt.figure(figsize = (12,6))
# plt.scatter(X1[0,:], X1[1,:], marker = '.', c = '#348ABD')
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
# plt.show()

# # Array to hold all grid points after transport
# X1 = np.zeros((2, Ny, Nx))

# # Transport parameters
# t_max = 32.0

# # Loop over grid and update all positions
# # This is where parallelisation would happen, since
# # each position is independent of all the others
# for i in range(Nx):
    # for j in range(Ny):
        # # Keep only the last position, not the entire trajectory
        # X1[:,j,i] = trajectory(XY[:,j,i], t_max, dt, rk4, f)[:,-1]

        
# # Make scatter plot to show all grid points
# fig = plt.figure(figsize = (12,6))
# plt.scatter(X1[0,:], X1[1,:], marker = '.', c = '#348ABD')
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
# plt.show()
