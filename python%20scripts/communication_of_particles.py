# example to run:
#mpiexec -n 2 python communication_of_particles.py
# where 2 is the number of processes

### PARAMETERS TO MANUALLY CHANGE:
# number of cells in each direction

import numpy as np
from mpi4py import MPI
from matplotlib import pyplot as plt
#plt.style.use('bmh')

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
mpi_size = comm.Get_size()

# initializing spatial properties
x_start = 0
x_end = 1
y_start = 0
y_end = 0 # 1D example

x_len = x_end - x_start
y_len = y_end - y_start

# initializing "processes/rank" properties
# TODO: implement "find biggest factor"-function or another dynamical function to determine numer of cells in each direction

###
# number of cells in each direction (0 in z direction at first draft/prototype)
cells_dir_n = 8

cell_x_n = cells_dir_n

#cell_y_n = cells_dir_n
#cell_n = cell_x_n*cell_y_n
# for 1D example/prototype:
cell_y_n = 0
cell_n = cell_x_n

# Particle "object" is defined in several arrays
# TODO: define a "particle-class/tuple"?
# particle_id is bound to the index in the x/y-arrays

# total number of particles
particle_n = 10
particle = np.arange(particle_n)
particle_x = np.linspace(x_start, x_end, particle_n, endpoint=False)
particle_y = np.linspace(y_start, y_end, particle_n, endpoint=False)

# function to find which process/rank of cell
### this function determines how the cells are distributed to ranks
### discussion: how to do this distribution
def find_rank_from_cell(cell_id):
    return (cell_id % mpi_size)

# function to find which cell of position
### this function determines how the cells are distributed geometric
### discussion: how to do this distribution
def find_cell_from_position(x):
    return np.floor(((x - x_start)/(x_len))*(cell_x_n)) # for 1D

# function to find which position of particle
# in 2D, return a rank 2 array?
def find_pos_from_particle(particle_id):
    return particle_x[particle_id] # for 1D
    #return (particle_x[particle_id], particle_y[particle_id])
	
### code to run in each rank

# local array to show how many particles should be sent from one rank to the others
# rows represent particles sent FROM rank = row number (0 indexing)
# column represent particles sent TO rank = row number (0 indexing)

# function to fill out the array showing number of particles need to be sent from a given rank given the local particles there
# local particles are the particles who belonged to the rank before the transport of particles.
# some of the particles may have to been moved to a new rank if they have been moved to a cell belonging to a new rank
def fill_send_n_array(rank, particle_local):
    send_n_array = np.zeros((mpi_size, mpi_size))
    for p in particle_local:
        particle_rank = find_rank_from_cell(find_cell_from_position(find_pos_from_particle(p)))
        # check if the cell the particle is in the same rank
        if particle_rank != rank:
            send_n_array[int(rank)][int(particle_rank)] = send_n_array[int(rank)][int(particle_rank)] + 1
			# converted indices to int to not get "deprecation warning"
    return send_n_array
	
# particles in local array AFTER transport "input" from transport "function"
# here we choose an arbitary* subsset of particles to assign to each rank
# *) arbitary: assign each n-th particle to a rank, where n = number of ranks

# array to show the particles in each rank, for debugging/help
particle_rank = np.zeros((mpi_size,0), dtype = int)

particle_local = particle[rank::mpi_size]
particle_n_local = particle_local.size

send_n_local = fill_send_n_array(rank, particle_local) # (sendbuf)

# communication
# ALL nodes receives results with a collective "Allreduce"
# mpi4py requires that we pass numpy objects.

send_n_global = np.zeros((mpi_size, mpi_size)) # recvbuf
comm.Allreduce(send_n_local , send_n_global , op=MPI.SUM)

print("rank:", rank)
#print("local_array:\n", local_array)
print("local particles:", particle_local)
print("belongs to ranks:", find_rank_from_cell(find_cell_from_position(find_pos_from_particle(particle_local))))
print("\n")

if rank == 0:
	#print("particles belongs to ranks (after transport):")
	#for i in range(mpi_size):
	print("global_array:\n", send_n_global)

### exchange/sending of particles ###
# method 1: array with shape (mpi_size, max_particle_send)
particle_send_local = np.zeros(mpi_size, np.amax(send_n_global))


