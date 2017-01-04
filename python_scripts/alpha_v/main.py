# main file for Python parallell particle transport

# Simen Mikkelsen, 2016-11-19
# main.py

# example to run:
"""
mpiexec -n 4 python main.py
"""
# where 4 is the number of processes
import mpi4py.MPI as MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
mpi_size = comm.Get_size()

from time import time

comm.Barrier()
if rank == 0:
    tic = time()


#import matplotlib.pyplot as plt
#plt.style.use('bmh')

import os
import sys
# import from other files
import matplotlib
matplotlib.use('agg')
import numpy as np
import IO #save_array_binary_file, load_array_binary_file, create_grid_of_particles, save_grid_of_particles
import transport #
import plot #
import communication #
import debugging as db

# temp functions

## VARIABLES start
# particle_n, number of particles is defined directly in IO-function input

# integrator timestep
dt   = 0.005
# number of timesteps between each communication event
Ndt = int(0.5/dt)

# simulation
# start time
t_0 = 0
# end time
t_max = 3

## VARIABLES end

print('\nrank', rank)
print('total number of ranks', mpi_size)

# get initial positions, from file or random or other
# option 1: create from function
#ids, active, XY = IO.create_grid_of_particles(N = 10**2, w = 0.1)

# option 2: load from file
# set all particles initially to rank 0, so the other ranks will load empty arrays
# id_init, active_init, XY_init = IO.create_grid_of_particles(N = 10**2, w = 0.1)
# if rank == 0:
    # IO.save_grid_of_particles(id_init, active_init, XY_init, t_0, rank, input = True)
# else:
    # IO.save_empty_grid_to_input(t_0, rank)

# then load the particle grid for the given rank

ids, active, XY = IO.load_grid_of_particles(rank, time = 0, input = True)
if rank == 0:
    Nparticles = np.sum(active)

# start at initial time
t = t_0
IO.save_grid_of_particles(ids, active, XY, t, rank)
#plot.plot(rank, XY[:,active], t, dt)

# main loop
print('\nstart time = %s' % t)

# Communicating, to avoid bad load balancing at the beginning
print('This is rank %s, communicating before mainloop' % (rank))
ids, XY, active = communication.exchange(comm, mpi_size, rank, ids, XY[0,:], XY[1,:], active)

while t + dt <= t_max:
    # Take Ndt timesteps
        
    # only take active particles into transport function
    # non-active particles should be in the end of the arrays,
    # so if we want we can use the index as XY[:,:np.sum(active)] instead
    print('This is rank %s, transporting %s particles' % (rank, np.sum(active)))
    sys.stdout.flush()
    XY[:,active], t = transport.transport(XY[:,active], active, t, Ndt, dt)
    #plot.plot(rank, XY[:,active], t, dt, name = '_after_transport-before_comm_')
    #t += dt # this increment is returned from transport-funcion
    ############
    # Then communicate
    '''
    # all variables taken in by exchange() are local variables for the given rank (except mpi_size)
    def exchange(communicator,
                mpi_size,
                rank,
                particle_n,
                particle_id,
                particle_x,
                particle_y,
                particle_active):
                
            return (particle_id,
                particle_x,
                particle_y,
                particle_active)
    '''
    
    print('This is rank %s, ready for communication' % (rank))
    sys.stdout.flush()
    ids, XY, active = communication.exchange(comm, mpi_size, rank, ids, XY[0,:], XY[1,:], active)
    
    # Then calculate concentration
    #############
    #plot.plot(rank, XY[:,active], t, dt)
    IO.save_grid_of_particles(ids, active, XY, t, rank)
    # Insert barrier here
    comm.Barrier()
    #plot.plot_from_files(t, mpi_size, dt)
    print('\nt = %s' % t)


timefilename = 'timing.txt'
comm.Barrier()
if rank == 0:
    toc = time()
    if not os.path.exists(timefilename):
        timefile = open(timefilename, 'w')
        timefile.write('Nparticles\tNcells\tNranks\tNcomm\tTmax\tdt\tTime\n')
        timefile.close()
    with open(timefilename, 'a') as timefile:
        Ncomm = 1 + int(t_max / dt / Ndt)
        timefile.write('%s\t%s\t%s\t%s\t%.5f\t%.5f\t%.5f\n' % (Nparticles, communication.cell_x_n, mpi_size, Ncomm, t_max, dt, toc - tic))


# IO-test:
    #a = [0, 1, 2, 3, 4]
    #name = "test"
    #a = IO.load_array_binary_file(name, 1, rank)
    #IO.save_array_binary_file(a, name, 2, rank)
