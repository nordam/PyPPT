# script to test transport of particles in a veolicty field
# the particles (properties) are loaded from binary python files

# Simen Mikkelsen, 2016-11-15

# example to run:
"""
mpiexec -n 1 python transport_of_particles.py
"""
# where 1 is the number of processes

import numpy as np

from mpi4py import MPI
import os # for IO-paths
import matplotlib.pyplot as plt
plt.style.use('bmh')

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
mpi_size = comm.Get_size()

### INITIALIZING global parameters/properties

## time and parameters of current veolicity field
    
t_0 = 0.0
#t_plot = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5]
t_plot = [0, 5]
A = 0.1
e = 0.25 # epsilon
w = 1 # omega

## transport parameters
t_max = 5
# time step in Runge-Kutta integrator
dt   = 0.5
# multiplication factor to compare different time steps 
dt_factor = 10

## grid of points
# specify resolution
Nx = 10
Ny = 10
N = Nx*Ny
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

# IO
dir = os.path.dirname(os.path.realpath(__file__))
IO_path = os.path.join(dir, 'simulation','files')
IO_path_init = os.path.join(dir, 'simulation','rectangle')

file_extension = '.npy'

file_path_id        = os.path.join(IO_path, 'particle_id' + str(rank))
file_path_x         = os.path.join(IO_path, 'particle_x' + str(rank))
file_path_y         = os.path.join(IO_path, 'particle_y' + str(rank))
file_path_active    = os.path.join(IO_path, 'particle_active' + str(rank))

file_path_id_init       = os.path.join(IO_path_init, 'particle_id' + str(rank))
file_path_x_init        = os.path.join(IO_path_init, 'particle_x' + str(rank))
file_path_y_init        = os.path.join(IO_path_init, 'particle_y' + str(rank))
file_path_active_init   = os.path.join(IO_path_init, 'particle_active' + str(rank))

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
    # Parameters of the velocity field
    A = 0.1
    e = 0.25 # epsilon
    w = 1    # omega
    return doublegyre(X[0,:], X[1,:], t, A, e, w)

# 4th order Runge-Kutta integrator
# X0 is a two-component vector [x, y]
def rk4(X, t, dt, f):
    k1 = f(X,           t)
    k2 = f(X + k1*dt/2, t + dt/2)
    k3 = f(X + k2*dt/2, t + dt/2)
    k4 = f(X + k3*dt,   t + dt)
    return X + dt*(k1 + 2*k2 + 2*k3 + k4) / 6
	
# Function to calculate a trajectory from an
# initial position X0 at t = 0, moving forward
# until t = tmax, using the given timestep and
# integrator
def trajectory(X, tmax, dt, integrator, f):
    t    = 0
    # Number of timesteps
    Nt = int(tmax / dt)
    # Loop over all timesteps
    for i in range(1, Nt+1):
        X = integrator(X, t, dt, f)
        t += dt
    # Return entire trajectory
    return X
    
def grid_of_particles(N, w):
    # Create a grid of N evenly spaced particles
    # covering a square patch of width and height w
    # centered on the region 0 < x < 2, 0 < y < 1
    x  = np.linspace(1.0-w/2, 1.0+w/2, int(np.sqrt(N)))
    y  = np.linspace(0.5-w/2, 0.5+w/2, int(np.sqrt(N)))
    x, y = np.meshgrid(x, y)
    return np.array([x.flatten(), y.flatten()])

### INITIALIZING end

print("total number of ranks:", mpi_size, "\nrank:", rank, "\n")

# initializing of local particle arrays 
# the local particles are gathered from the local properties arrays, loaded from files

particle_id_local = np.load(file_path_id_init + file_extension)
particle_x_local = np.load(file_path_x_init + file_extension)
particle_y_local = np.load(file_path_y_init + file_extension)
particle_active_local = np.load(file_path_active_init + file_extension)

# Make a plot to confirm that this works as expected
fig = plt.figure(figsize  = (12,6))
plt.scatter(particle_x_local, particle_y_local, lw = 0, marker = '.', s = 1)
plt.xlim(0, 2)
plt.ylim(0, 1)

### plot particle rectangle before and after transport

X = grid_of_particles(N, 0.1)
# Make a plot to confirm that this works as expected
fig = plt.figure(figsize  = (12,6))
plt.scatter(X[0,:], X[1,:], lw = 0, marker = '.', s = 1)
plt.xlim(0, 2)
plt.ylim(0, 1)

XY = [particle_x_local, particle_y_local]
print("particles_XY:", XY)
XY = X
# Array to hold all grid points after transport
XY1 = np.zeros((2, N))

# Transport parameters
# in initializing

# Loop over grid and update all positions
# This is where parallelisation would happen, since
# each position is independent of all the others
# Keep only the last position, not the entire trajectory
#XY1 = trajectory(XY, t_max, dt, rk4, f)
XY1 = trajectory(XY, 5.0, 0.5, rk4, f)
print("grid_XY:", XY)
print("XY1:",XY1)

# Make scatter plot to show all grid points
fig = plt.figure(figsize = (12,6))
plt.scatter(XY1[0,:], XY1[1,:], lw = 0, marker = '.', s = 1)
plt.xlim(0, 2)
plt.ylim(0, 1)
plt.show()




# # Create lists of all x and y positions
# X  = np.linspace(particles_x_min, particles_x_max, Nx)
# Y  = np.linspace(particles_y_min, particles_y_max, Ny)

# # load from file

# # Array to hold all grid points
# XY = np.zeros((2, Ny, Nx))

# # Use meshgrid to turn lists into rank 2 arrays
# # of x and y positions
# XY[:] = np.meshgrid(X, Y)

# # subplots
# fig = plt.figure(figsize = (6,6))

# # start positions and transportation

# for i in range(np.size(t_plot)):
    # plt.subplot(2, np.size(t_plot)/2, i+1)
    # t = t_plot[i]
    
    # # Array to hold all grid points after transport
    # XY_t = np.zeros((2, Ny, Nx))

    # # Loop over grid and update all positions
    # # This is where parallelisation would happen, since
    # # each position is independent of all the others
    # for i in range(Nx):
        # for j in range(Ny):
            # # Keep only the last position, not the entire trajectory
            # XY_t[:,j,i] = trajectory(XY[:,j,i], t, dt, rk4, f)[:,-1]

            
    # # Make scatter plot to show all grid points
    # plt.scatter(XY_t[0,:], XY_t[1,:], marker = '.', c = '#348ABD')
    # # add text showing time, and set plot limits
    # plt.text(1.65, 0.9, '$t = %s$' % t, size = 36)
    # plt.xlim(x_min, x_max)
    # plt.ylim(y_min, y_max)

# plt.show()