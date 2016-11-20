'''
program to transport particles and then communicate/exchange particles (properties of particles) with other ranks

Simen Mikkelsen, 2016-11-15

example to run with 4 process:
mpiexec -n 4 python transport_and_communication_of_particles.py
''' 

import numpy as np
import os # for IO-paths

from mpi4py import MPI
from matplotlib import pyplot as plt

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
mpi_size = comm.Get_size()
request = MPI.Request

## INITIALIZING of global parameters/properties

# time and parameters of current veolicity field  
t_0 = 0.0
A = 0.1
e = 0.25 # epsilon
w = 1 # omega

# transport parameters
t_max = 2
dt   = 0.1 # time step in Runge-Kutta integrator

# spatial properties
x_start = 0
x_end = 2
y_start = 0
y_end = 1

x_len = x_end - x_start
y_len = y_end - y_start

# 'processes/rank'-properties
# TODO: implement 'find biggest factor'-function or another dynamical function to determine number of cells in each direction

# number of cells in each direction (only divide x-direction at first draft/prototype)
cells_dir_n = 16
cell_x_n = cells_dir_n

#cell_y_n = cells_dir_n
#cell_n = cell_x_n*cell_y_n
cell_y_n = 0
cell_n = cell_x_n

# scaling factor when expanding/scrinking local arrays
scaling_factor = 1.1 ## variable
shrink_if = 1/(scaling_factor**3)

# the particles are defined with its properties in several arrays
# id, x-pos, y-pos, active-status

# communication
# numbers of tags
tag_n = 3
# tags: id, x, y
# buffer_overhead multiplication factor
buffer_overhead = 5

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

## INITIALIZING end

# function to find the corresponding rank of a cell
# this function determines how the cells are distributed to ranks
### TODO: discussion: how to do this distribution
def find_rank_from_cell(cell_id):
    return int(cell_id % mpi_size)

# function to find the corre cell of a position
# this function determines how the cells are distributed geometric
### TODO: discussion: how to do this distribution
def find_cell_from_position(x, y):
    return int(((x - x_start)/(x_len))*(cell_x_n)) # for 1D

# function to reallocate active particles to the front of the local arrays
# active_n = number of active particles after deactivation of particles sent to another rank, but before receiving.
# aka. particles that stays in its own rank
def move_active_to_front(particle_id_local, particle_x_local, particle_y_local, particle_active_local, active_n):
    particle_id_local[:active_n] = particle_id_local[particle_active_local]
    particle_x_local[:active_n] = particle_x_local[particle_active_local]
    particle_y_local[:active_n] = particle_y_local[particle_active_local]
    # set the corresponding first particles to active, the rest to false
    particle_active_local[:active_n] = True
    particle_active_local[active_n:] = False
    ### TODO: return the arrays here?
    return

## code to run in each rank:

# initializing of local particle arrays 
# the local particles are gathered from the local properties arrays, loaded from files

particle_id_local = np.load(file_path_id_init + file_extension)
particle_x_local = np.load(file_path_x_init + file_extension)
particle_y_local = np.load(file_path_y_init + file_extension)
particle_active_local = np.load(file_path_active_init + file_extension)

# length of particle arrays (local arrays)
particle_n_local = particle_id_local.size

print("\n")
print('rank:', rank)
print('length of local arrays:', particle_n_local)
print("active particles:", particle_active_local.sum())
#print("local particles:", particle_id_local)
#print("active particles:", particle_active_local*1) # *1 to turn the output into 0 and 1 instead of False and True
#print("belongs to rank:", find_rank_from_cell(find_cell_from_position(particle_x_local, particle_y_local)))
#print("x-positions:", particle_x_local)
#print("y-positions:", particle_y_local)

# transport particles:
### TODO: implement doublegyre function to transport particles

# send_n_array: array to show how many particles should be sent from one rank to the others
# filled out locally in each rank, then communicated to all other ranks
# rows represent particles sent FROM rank = row number (0 indexing)
# column represent particles sent TO rank = row number (0 indexing)
# function to fill out the array showing number of particles need to be sent from a given rank given the local particles there
# local particles are the particles who belonged to the rank before the transport of particles.
# some of the particles may have to been moved to a new rank if they have been moved to a cell belonging to a new rank
# send_to: array to show which rank a local particle needs to be sent to. or -1 if it should stay in the same rank
def fill_communication_arrays(rank, particle_n_local, mpi_size, particle_x_local, particle_y_local):
    # reset arrays telling which particles are to be sent
    send_to = np.zeros(particle_n_local, dtype=int) # local
    send_to[:] = -1
    send_n_array = np.zeros((mpi_size, mpi_size), dtype=int)
    for i in range(particle_n_local):
        # only check if the particle is active
        if particle_active_local[i]:
            # find the rank of the cell of which the particle (its position) belongs to
            particle_rank = find_rank_from_cell(find_cell_from_position(particle_x_local[i], particle_y_local[i]))
            # if the particle's new rank does not equal the current rank (for the given process), it should be moved
            if particle_rank != rank:
                send_n_array[int(rank)][int(particle_rank)] = send_n_array[int(rank)][int(particle_rank)] + 1
                send_to[i] = particle_rank
                # converted indices to int to not get 'deprecation warning'
    return send_to, send_n_array

send_to_local, send_n_local = fill_communication_arrays(rank, particle_n_local, mpi_size, particle_x_local, particle_y_local)
# sendbuffers: sendbuf 1, sendbuf 2

# communication
# all nodes receives results with a collective 'Allreduce'
# mpi4py requires that we pass numpy objects.

# receive buffer (recvbuf)
send_n_global = np.zeros((mpi_size, mpi_size), dtype=int)
comm.Allreduce(send_n_local , send_n_global , op=MPI.SUM)

# print global array
print('\nglobal_array:\n', send_n_global)    

## exchange/sending of particles
# list of arrays for communication
# initializing 'communication arrays': send_*** and recv_***
# send_**: list of arrays to hold particles that are to be sent from a given rank to other ranks, where row number corresponds to the rank the the particles are send to
# recv_**: list of arrays to hold particles that are to be received from to a given rank from other ranks, where row number corresponds to the rank the particles are sent from

send_id_local = []
send_x_local = []
send_y_local = []
recv_id_local = []
recv_x_local = []
recv_y_local = []

for irank in range(mpi_size):
    # find number of particles to be received from irank (sent to current rank)
    Nrecv = send_n_global[irank, rank]
    # append recv_id_local with the corresponding number of elements
    recv_id_local.append([None]*Nrecv)
    recv_x_local.append([None]*Nrecv)
    recv_y_local.append([None]*Nrecv)
        
    # find number of particles to be sent to irank (from current rank)
    Nsend = send_n_global[rank, irank]
    # append send_id_local with the corresponding number of elements
    send_id_local.append([None]*Nsend)
    send_x_local.append([None]*Nsend)
    send_y_local.append([None]*Nsend)
    
# counter to get position in send_**_local for a particle to be sent
send_count = np.zeros(mpi_size, dtype=int)

# iterate over all local particles to allocate them to send_** if they belong in another rank
for i in range(particle_n_local):
    # if particle is active (still a local particle) and should be sent to a rank (-1 means that the particle already is in the correct rank)
    if (particle_active_local[i] and send_to_local[i] != -1):
        # fill the temporary communication arrays (send_**) with particle and it's properties
        send_id_local[send_to_local[i]][send_count[send_to_local[i]]] = i
        send_x_local[send_to_local[i]][send_count[send_to_local[i]]] = particle_x_local[i]
        send_y_local[send_to_local[i]][send_count[send_to_local[i]]] = particle_y_local[i]
        
        # deactivate sent particle
        particle_active_local[i] = False
        # increment counter to update position in temporary communication arrays (send_**)
        send_count[send_to_local[i]] = send_count[send_to_local[i]] + 1

## point-to-point communication
# communicate with all other ranks

# must convert objects to be communicated to byte-like objects (numpy-objects)
# this is not done before because np.ndarrays does not support a 'list of arrays' if not each array have equal dimensions
send_id_np = np.array(send_id_local)
recv_id_np = np.array(recv_id_local)
send_x_np = np.array(send_x_local)
recv_x_np = np.array(recv_x_local)
send_y_np = np.array(send_y_local)
recv_y_np = np.array(recv_y_local)

# send_id_np_buf = np.array(send_id_local_buf)
# recv_id_np_buf = np.array(recv_id_local_buf)
# send_x_np_buf = np.array(send_x_local_buf)
#recv_x_np_buf = np.array(recv_x_local_buf)
# send_y_np_buf = np.array(send_y_local_buf)
# recv_y_np_buf = np.array(recv_y_local_buf)

# requests to be used for non-blocking send and receives
# first only for ID, then for x and y
#send_request = np.zeros(0)
#recv_request = np.zeros(0)
send_request_id = [None]*mpi_size
send_request_x = [None]*mpi_size
send_request_y = [None]*mpi_size

# sending

for irank in range(mpi_size):
    if (irank != rank):
        # number of particles rank sends to irank
        Nsend = send_n_global[rank, irank] 
        # only receive if there is something to recieve
        if (Nsend > 0):
            #print('rank:', rank, 'sending', Nsend, 'particles to', irank)
            # use tags to separate communication of different arrays/properties
            # tag uses 1-indexing so there will be no confusion with the default tag = 0
            send_request_id[irank]  = comm.isend(send_id_np[irank][0:Nsend], dest = irank, tag = 1) # could have written [:] insted of [0:Nsend]
            send_request_x[irank]   = comm.isend(send_x_np[irank][0:Nsend], dest = irank, tag = 2)
            send_request_y[irank]   = comm.isend(send_y_np[irank][0:Nsend], dest = irank, tag = 3)
            #print('sending:', send_id_np[irank][0:Nsend])#, send_x_np[irank][0:Nsend])
#print('send_id_np:', send_id_np)

# receiving

recv_request_id = [None]*mpi_size
recv_request_x = [None]*mpi_size
recv_request_y = [None]*mpi_size

for irank in range(mpi_size):
    if (irank != rank):
        # number of particles irank sends to rank (number of particles rank recieves from irank)
        Nrecv = send_n_global[irank, rank]
        # only receive if there is something to recieve
        if (Nrecv > 0):
            #print('rank:', rank, 'receiving', Nrecv, 'particles from', irank)
            # use tags to separate communication of different arrays/properties
            # tag uses 1-indexing so there will be no confusion with the default tag = 0
            buf_id = np.zeros(Nrecv*buffer_overhead, dtype = np.int)
            buf_x = np.zeros(Nrecv*buffer_overhead, dtype = np.float64)
            buf_y = np.zeros(Nrecv*buffer_overhead, dtype = np.float64)
            
            # tag uses 1-indexing so there will be no confusion with the default tag = 0
            recv_request_id[irank]  = comm.irecv(buf = buf_id, source = irank, tag = 1)
            recv_request_x[irank]   = comm.irecv(buf = buf_x, source = irank, tag = 2)
            recv_request_y[irank]   = comm.irecv(buf = buf_y, source = irank, tag = 3)
            
            #recv_id_np[irank][0:Nrecv] = comm.recv(buf=None, source = irank, tag = 1)
            #recv_x_np[irank][0:Nrecv] = comm.recv(buf=None, source = irank, tag = 2)
            #recv_y_np[irank][0:Nrecv] = comm.recv(buf=None, source = irank, tag = 3)
            #print('receiving:', recv_id_np[irank][0:Nrecv])#, send_x_np[irank][0:Nsend])

#print("recv_request_id:", recv_request_id)
#print("send_request_id:", send_request_id)

# obtain data from completed requests
# only at this step is the data actually returned
for irank in range(mpi_size):
    if irank != rank:
        # if there is something to receive
        if send_n_global[irank, rank] > 0: # Nrexv > 0
            recv_id_np[irank][:] = recv_request_id[irank].wait()
            recv_x_np[irank][:] = recv_request_x[irank].wait()
            recv_y_np[irank][:] = recv_request_y[irank].wait()
        
#print('recv_id_np WAITED:', recv_id_np)
#print("recv_x_np WAITED:", recv_x_np)
#print("recv_y_np WAITED:", recv_y_np)

# make sure this rank does not exit until sends have completed
for irank in range(mpi_size):
    if irank != rank:
        # if there is something to send
        if send_n_global[rank, irank] > 0: # Nsend > 0
            send_request_id[irank].wait()
            send_request_x[irank].wait()
            send_request_y[irank].wait()

# total number of received and sent particles
# total number of active particles after communication
sent_n      = np.sum(send_n_global, axis = 1)[rank]
received_n  = np.sum(send_n_global, axis = 0)[rank]
active_n    = np.sum(particle_active_local)

print("sent_n:", sent_n)
print("received_n:", received_n)
print("active_n:", active_n)

# move all active particles to front of local arrays
if (active_n > 0):
    #print('\nmove active particles to front')
    #print('particle_id_local:', particle_id_local)
    #print('particle_active_local before movement to front:', particle_active_local)
    move_active_to_front(particle_id_local, particle_x_local, particle_y_local, particle_active_local, active_n)

new_length = particle_n_local
### TODO: add ceil/floor directly in if-check?
# current scaling factor = 1.25
# check if local arrays have enough free space, if not, allocate a 'scaling_factor' more than needed
if (active_n + received_n > particle_n_local):
    new_length = int(np.ceil((active_n + received_n)*scaling_factor))
    # if new length is not equal old length: resize all local arrays
    if new_length != particle_n_local:
        #print('extending arrays to new length:', new_length)
        particle_active_local.resize(new_length)#, refcheck = False)
        # with .resize-method, missing/extra/new entries are filled with zero = false
        particle_id_local.resize(new_length)#, refcheck = False) # refcheck = True by default
        particle_x_local.resize(new_length)
        particle_y_local.resize(new_length)

# check if local arrays are bigger than needed (with a factor: shrink_if = 1/scaling_factor**3)
# old + new particles < shrink_if*old_size
# if they are, shrink them with a scaling_factor
if (active_n + received_n < shrink_if*particle_n_local):
    new_length = int(np.ceil(particle_n_local/scaling_factor))
    # if new length is not equal old length: resize all local arrays
    if new_length != particle_n_local:
        #print('shrinking arrays to new length:', new_length)
        particle_active_local.resize(new_length)
        particle_id_local.resize(new_length)#, refcheck = False) # refcheck = True by default
        particle_x_local.resize(new_length)
        particle_y_local.resize(new_length)
        
# add the received particles to local arrays     

# unpack (hstack) the list of arrays, (ravel/flatten can not be used for dtype=object)
#print('_flattend_ id:', np.hstack(recv_id_np))

print('\nnew length of local arrays:', new_length)
if received_n > 0:
    particle_id_local[active_n:active_n+received_n] = np.hstack(recv_id_np)
    particle_x_local[active_n:active_n+received_n] = np.hstack(recv_x_np)
    particle_y_local[active_n:active_n+received_n] = np.hstack(recv_y_np)
# set the received particles to active
    particle_active_local[active_n:active_n+received_n]  = [True]*received_n
# print the new values
    print("new local particles:", particle_id_local)
    print("new active particles:", particle_active_local*1) # *1 to turn the output into 0 and 1 instead of False and True
else:
    print("\nno received particles")
    
## save files

np.save(file_path_id, particle_id_local)
np.save(file_path_x, particle_x_local)
np.save(file_path_y, particle_y_local)
np.save(file_path_active, particle_active_local)

#comm.barrier()
#comm.wait()