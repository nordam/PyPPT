# script with functions to use in main
# this script handle the "transport of particles"-part

# Simen Mikkelsen, 2016-11-19

# transport.py
'''
def timestep(X, dt, t):
    # Use double gyre and RK4 to calculate new functions
    return X
'''
 
import numpy as np


##  INITIALISING start

## INITIALISING end

## VARIABLES start

## VARIABLES end

## FUNCTIONS start

# doublegyre velocity field
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
def f(XY, t):
    # Parameters of the velocity field
    A = 0.1
    e = 0.25 # epsilon
    w = 1    # omega
    return doublegyre(XY[0,:], XY[1,:], t, A, e, w)

# 4th order Runge-Kutta integrator. XY is a two-component vector [x, y]
def rk4(XY, t, dt, f):
    k1 = f(XY,           t)
    k2 = f(XY + k1*dt/2, t + dt/2)
    k3 = f(XY + k2*dt/2, t + dt/2)
    k4 = f(XY + k3*dt,   t + dt)
    return XY + dt*(k1 + 2*k2 + 2*k3 + k4) / 6
    
# function to calculate a trajectory from an initial position XY0 at t = 0,
# moving forward until t = t_max, using the given timestep and integrator
# XY is a two-component vector [x, y]
def trajectory(XY, current_time, Ndt, dt, integrator, f):
    t = current_time
    # number of timesteps
    #Nt = int(t_max/dt)
    # loop over all timesteps
    #for i in range(1, Ndt+1):
    for i in range(Ndt):
    ## TODO: what's the difference between the last line and the line over it?
        XY = integrator(XY, t, dt, f)
        t += dt
    # return entire trajectory and current time
    return XY, t
    
def transport(XY0, particle_active, current_time, Ndt, dt):
    ### TODO: only transport active particles?
    ### for example set non-active particle coordinates to (0,0)?
    ###  or remove them?
    ### or do not send them to trajectory(), thus do not change them?
    ### conclusion:
    ###             assumes XY0 har all active particles firs 
    ###             transport only the active particles
    ###             take in active_n instead of whole array?
    # XY0 is a two-component vector [x, y]
    # loop over grid and update all positions
    # this is where parallelisation would happen, since each position is independent of all the others
    
    # array to hold all grid points after transport
    #XY1 = np.zeros((2, np.size(particle_active)))
    # keep only the last position, not the entire trajectory
    XY, t = trajectory(XY0, current_time, Ndt, dt, rk4, f)
    return XY, t

## FUNCTIONS end