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

#import matplotlib.pyplot as plt
#plt.style.use('bmh')

# import from other files
import numpy as np
import IO #save_array_binary_file, load_array_binary_file, create_grid_of_particles, save_grid_of_particles
import transport #
import plot #
import communication #

# temp functions

## VARIABLES start
# particle_n, number of particles is defined directly in IO-function input

# integrator timestep
dt   = 0.5
# number of timesteps between each communication event
# now: plot each time, TODO: implement exchange/communications
Ndt = 10

# simulation
# start time
t_0 = 0
# end time
t_max = 10

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

ids, active, XY = IO.load_grid_of_particles(rank, time = 0)

# start at initial time
t = t_0
IO.save_grid_of_particles(ids, active, XY, t, rank)
plot.plot(rank, XY, t, dt, active)

# main loop
while t < t_max:
    print('\nt = %s' % t)
    # Take Ndt timesteps
    XY, t = transport.transport(XY[:,active], active, t, Ndt, dt)
    plot.plot(rank, XY, t, dt, active, name = 'after_transport-before_comm')
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
    ids, XY, active = communication.exchange(comm, mpi_size, rank, ids, XY[0,:], XY[1,:], active)

    # Then calculate concentration
    #############
    plot.plot(rank, XY, t, dt, active)
    IO.save_grid_of_particles(ids, active, XY, t, rank)

#XY1, t = transport.transport(XY, particle_n, t, t_max, dt)
#plot.plot(XY1, t)
#IO.save_grid_of_particles(id, XY1, t, rank)


# IO-test:
    #a = [0, 1, 2, 3, 4]
    #name = "test"
    #a = IO.load_array_binary_file(name, 1, rank)
    #IO.save_array_binary_file(a, name, 2, rank)