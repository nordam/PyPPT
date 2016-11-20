# main file for exchange_of_particles-procedure

# Simen Mikkelsen, 2016-11-19
# main.py

# example to run:
"""
mpiexec -n 1 python main.py
"""
# where 4 is the number of processes
import numpy as np

import mpi4py.MPI as MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
mpi_size = comm.Get_size()

#import matplotlib.pyplot as plt
#plt.style.use('bmh')

# import from other files
import IO #save_array_binary_file, load_array_binary_file, create_grid_of_particles, save_grid_of_particles
import transport #
import plot

# temp functions

## VARIABLES start
# number of particles
particle_n = 10**2

# integrator timestep
dt   = 0.5
# number of timesteps between each communication event
# now: plot each time, TODO: implement exchange/communications
Ndt = 2

# simulation
# start time
t_0 = 0
# end time
t_max = 50

## VARIABLES end

print("rank", rank)

# get initial positions, from file or random or other
# create from function
id, XY = IO.create_grid_of_particles(particle_n, w = 0.1)

# start at initial time
t = t_0
IO.save_grid_of_particles(id, XY, t, rank)
plot.plot(XY, t)

# main loop
while t < t_max:
    print('t = %s' % t)
    # Take Ndt timesteps
    #for i in range(Ndt):
        ## TODO: break for loop if t > t_max
        #print('for loop, i:', i)
    XY, t = transport.transport(XY, particle_n, t, Ndt, dt)
         
        #t += dt # this increment is returned from transport-funcion
    # Then communicate
    #X = exchange(X)
    # Then calculate concentration
    plot.plot(XY, t)
    IO.save_grid_of_particles(0, XY, t, rank)

#XY1, t = transport.transport(XY, particle_n, t, t_max, dt)
#plot.plot(XY1, t)
#IO.save_grid_of_particles(id, XY1, t, rank)


# IO-test:
    #a = [0, 1, 2, 3, 4]
    #name = "test"
    #a = IO.load_array_binary_file(name, 1, rank)
    #IO.save_array_binary_file(a, name, 2, rank)