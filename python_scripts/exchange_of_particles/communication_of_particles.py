# example to run:
"""
mpiexec -n 4 python communication_of_particles.py
"""
# where 4 is the number of processes

import numpy as np

from mpi4py import MPI
from matplotlib import pyplot as plt
#plt.style.use('bmh')

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
mpi_size = comm.Get_size()

### INITIALIZING of global parameters/properties
# initializing spatial properties
x_start = 0
x_end = 1
y_start = 0
y_end = 0 # 1D example

x_len = x_end - x_start
y_len = y_end - y_start

# initializing "processes/rank" properties
# TODO: implement "find biggest factor"-function or another dynamical function to determine number of cells in each direction

# number of cells in each direction (only divide x-direction at first draft/prototype)
cells_dir_n = 8
cell_x_n = cells_dir_n

#cell_y_n = cells_dir_n
#cell_n = cell_x_n*cell_y_n
# for 1D example/prototype:
cell_y_n = 0
cell_n = cell_x_n

# scaling factor when expanding/scrinking local arrays
scaling_factor = 1.1 ## variable
shrink_if = 1/(scaling_factor**3)

# the particle "object" is defined in several arrays
# particle_id is bound to the index in the x/y-arrays
# total number of particles
particle_n = 10
# array with particle IDs
particle = np.arange(particle_n)
particle_x = np.linspace(x_start, x_end, particle_n, endpoint=False)
particle_y = np.linspace(y_start, y_end, particle_n, endpoint=False)

##### INITIALIZING end

# function to find which process/rank of cell
### this function determines how the cells are distributed to ranks
### discussion: how to do this distribution
def find_rank_from_cell(cell_id):
    return (cell_id % mpi_size)

# function to find which cell of position
### this function determines how the cells are distributed geometric
### discussion: how to do this distribution
def find_cell_from_position(x):
    return (np.floor(((x - x_start)/(x_len))*(cell_x_n))) # for 1D

# function to reallocate active particles to the front of "active"-array
# active_n = "number of active particles after sending, before receiving. aka. particles that stays in its own rank"
def move_active_to_front(particle_local, particle_x_local, particle_active_local, active_n):
    print("particle_local:", particle_local)
    print("particle_active_local:", particle_active_local)
    print("active_n:", active_n)
    particle_local[:active_n] = particle_local[particle_active_local]
    particle_x_local[:active_n] = particle_x_local[particle_active_local]
    # set the corresponding first particles to active, the rest to false
    particle_active_local[:active_n] = True
    particle_active_local[active_n:] = False
    return
    
# function to find which position of particle
# no need for this function, can just call "particle_x_local = particle_x[particle_local]"
# in 2D, return a rank 2 array?
#def find_pos_from_particle(particle_id):
#    return particle_x[particle_id] # for 1D
    #return (particle_x[particle_id], particle_y[particle_id])
    
### code to run in each rank:

## initializing of local particle array
# particles in local array represents positions AFTER transport with a "transport function"
## TODO: implement doublegyre function to transport particles
# here we choose an arbitary* subset of particles to assign to each rank
# *) arbitary: assign each n-th particle to the same rank, where n = number of ranks
# particle 0 to rank 0, particle 1 to rank 1, (...), particle n to rank 0, particle n+1 to rank 1 etc.

particle_local = np.zeros(np.size(particle[rank::mpi_size]), dtype = int) # this size-method or ndarray.size?
particle_local[:] = particle[rank::mpi_size]
particle_n_local = particle_local.size # or this size-method?

particle_x_local = particle_x[particle_local]
### particle_local_position = find_pos_from_particle(particle_local) # old way via function

particle_active_local = np.zeros(particle_n_local, dtype=bool) # local
# set all particles to active
particle_active_local[:] = True

### in the iterations after the first one, the local values are gathered from the last iteration

# local array to show how many particles should be sent from one rank to the others
# rows represent particles sent FROM rank = row number (0 indexing)
# column represent particles sent TO rank = row number (0 indexing)
# function to fill out the array showing number of particles need to be sent from a given rank given the local particles there
# local particles are the particles who belonged to the rank before the transport of particles.
# some of the particles may have to been moved to a new rank if they have been moved to a cell belonging to a new rank
def fill_communication_arrays(rank, particle_local):
    # reset arrays telling which particles are to be sent
    send_to = np.zeros(particle_n_local, dtype=int) # local
    send_to[:] = -1
    send_n_array = np.zeros((mpi_size, mpi_size), dtype=int)
    for i in range(particle_n_local):
    #for p in particle_local:
        # find the rank of the cell of which the particle position belongs to
        particle_rank = find_rank_from_cell(find_cell_from_position(particle_x_local[i]))
        # if the particle's new rank does not equal the current rank (for the given process), it should be moved
        if particle_rank != rank:
            send_n_array[int(rank)][int(particle_rank)] = send_n_array[int(rank)][int(particle_rank)] + 1
            send_to[i] = particle_rank
            # converted indices to int to not get "deprecation warning"
    return send_to, send_n_array

# array to show the particles in each rank, for debugging/help
##particle_rank = np.zeros((mpi_size,0), dtype = int)

send_to_local, send_n_local = fill_communication_arrays(rank, particle_local) # (sendbuf 1 and 2)

# communication
# ALL nodes receives results with a collective "Allreduce"
# mpi4py requires that we pass numpy objects.

send_n_global = np.zeros((mpi_size, mpi_size), dtype=int) # recvbuf
comm.Allreduce(send_n_local , send_n_global , op=MPI.SUM)
print("\n")
print("rank:", rank)
#print("particle_local:", particle_local)
#print("belongs to rank:", find_rank_from_cell(find_cell_from_position(particle_x_local)))
#print("will be sent to:", send_to_local, "(sent_to-array)")

if rank == 0:
    #print("particles belongs to ranks (after transport):")
    #for i in range(mpi_size):
    print("global_array:\n", send_n_global)

### exchange/sending of particles ###
## method 1: array with shape (mpi_size, max_particle_send)
#particle_send_local = np.zeros(mpi_size, np.amax(send_n_global))
#send_i = 0
# for p in particle_local:
    # particle_rank = find_rank_from_cell(find_cell_from_position(find_pos_from_particle(p)))
    # # check if the cell the particle is in the same rank
    # if particle_rank != rank:
        # particle_send_local[send_i] = p
        # send_i = send_i + 1
# print(rank, particle_send_local)

## method 2: list of arrays for communication
# initializing "communication arrays": send_*** and recv_***
# (Fortran code variable names: mpi_size = my_mpi_size, rank = my_mpi_rank)
# send_id: list of arrays to show which particles are to be sent from a given rank to other ranks, where row number corresponds to the rank the the particles are send to
# recv_id: list of arrays to show which particles are to be received from to a given rank from other ranks, where row number corresponds to the rank the particles are sent from

send_id_local = []
send_x_local = []
recv_id_local = []
recv_x_local = []

# total numbeer of received particles
received_n = np.sum(send_n_global, axis = 0)[rank] ## variable

for irank in range(mpi_size):
    # find number of particles to be received from irank (sent to current rank)
    Nrecv = send_n_global[irank, rank]
    # append recv_id_local with the corresponding number of elements
    recv_id_local.append([None]*Nrecv)
    recv_x_local.append([None]*Nrecv)
    
    # find number of particles to be sent to irank (from current rank)
    Nsend = send_n_global[rank, irank]
    # append send_id_local with the corresponding number of elements
    send_id_local.append([None]*Nsend)
    send_x_local.append([None]*Nsend)
    
# counter to get position in send_id_local for a particle to be sent
send_count = np.zeros(mpi_size, dtype=int)

# iterate over particle number (index or "local particle id") as now or: the particles in particle_local, or from the global particle id?
for i in range(particle_n_local):
    # if particle is active (still a local particle) and should be sent to a rank (-1 means that the particle already is in the correct rank)
    if (particle_active_local[i] and send_to_local[i] != -1):
        #print('rank:', rank, 'send_to:', send_to_local[i])
        # fill the temporary communication arrays (send_**) with particle and it's properties
        send_id_local[send_to_local[i]][send_count[send_to_local[i]]] = i
        send_x_local[send_to_local[i]][send_count[send_to_local[i]]] = particle_x_local[i]
        #send_id(send_to+1)%p(sendcount(send_to + 1)) = particles % id(ipart)
        #send_y(send_to+1)%p(sendcount(send_to + 1)) = particles % y(ipart)
        #send_z(send_to+1)%p(sendcount(send_to + 1)) = particles % z(ipart)
        
        # deactivate sent particle
        particle_active_local[i] = False
        # increment counter to update position in temporary communication arrays (send_**)
        send_count[send_to_local[i]] = send_count[send_to_local[i]] + 1
#print("send_id:" ,send_id_local)
#print("send_x:" ,send_x_local)
#print("\n")

# must convert objects to be communicated to byte-like objects (numpy-objects)
send_id_np = np.array(send_id_local)
recv_id_np = np.array(recv_id_local)
send_x_np = np.array(send_x_local)
recv_x_np = np.array(recv_x_local)

### point-to-point communication
# communicate with all other ranks
print("\ncommunication:")

# sending

for irank in range(mpi_size):
    if (rank != irank):
        # number of particles rank sends to irank
        Nsend = send_n_global[rank, irank] 
        # only receive if there is something to recieve
        if (Nsend > 0):
            print("rank:", rank, "sending", Nsend, "particles to", irank)
            # Use tags to separate communication
            tag = 1 # tag uses 1-indexing so there will be no confusion with the default tag = 0
            #mpi_recv_request_valid(tag, irank + 1) = .true.
            # first: try with blocking communicator
            #comm.recv(recv_id_np[irank][:], source = irank, tag = 1) # could have written [0:Nrecv] insted of all ([:])
            #comm.send(s[irank][0:Nsend], dest = irank, tag = 1)
            comm.send(send_id_np[irank][0:Nsend], dest = irank, tag = 1)
            comm.send(send_x_np[irank][0:Nsend], dest = irank, tag = 2)
            #print("sending:", send_id_np[irank][0:Nsend], send_x_np[irank][0:Nsend])
print("sendt:", send_id_np, send_x_np)

# receiving
## TODO: use recv-buffer?
for irank in range(mpi_size):
    if (rank != irank):
        # number of particles irank sends to rank (number of particles rank recieves from irank)
        Nrecv = send_n_global[irank, rank]
        # only receive if there is something to recieve
        if (Nrecv > 0):
            print("rank:", rank, "receiving", Nrecv, "particles from", irank)
            # Use tags to separate communication
            tag = 1 # tag uses 1-indexing so there will be no confusion with the default tag = 0
            # mpi_recv_request_valid(tag, irank + 1) = .true.
            # first: try with blocking communicator
            #comm.recv(recv_id_np[irank][:], source = irank, tag = 1) # could have written [0:Nrecv] insted of all ([:])
            #comm.recv(r[irank][0:Nrecv], source = irank, tag = 1)
            #print("RECV:", comm.recv(r, source = irank, tag = 1))
            #print("TEST recv")
            recv_id_np[irank][0:Nrecv] = comm.recv(buf=None, source = irank, tag = 1)
            recv_x_np[irank][0:Nrecv] = comm.recv(buf=None, source = irank, tag = 2)
print("received:", recv_id_np,"," , recv_x_np)

active_n = np.sum(particle_active_local)
    
#print("\n values before communication:")
#print("particle_n_local", particle_n_local)
#print("particle_local", particle_local)
#print("particle_x_local", particle_x_local)
#print("old particle_active_local", particle_active_local)
#print("active_n = number of old particles:", active_n)
#print("number of incoming particles:", received_n)
#print("length of local array:", particle_n_local)
#print("\n")

# local arrays managment
print("\nlocal arrays managment:")

### todo: add ceil/floor directly in if-check?
# current scaling factor = 1.25
# check if local arrays have enough free space, if not, allocate a "scaling_factor" more than needed
if (active_n + received_n > particle_n_local):
    new_length = int(np.ceil((active_n + received_n)*scaling_factor))
    print("new_length:", new_length)
    #add = np.ceil((active_n + received_n - particle_n_local)*scaling_factor)
    #print("add number of elements:", add)
    #print("TEST: it should equal:", new_length - particle_n_local)
        
    # if new length not equal old length
    # resize all local arrays
    if new_length != particle_n_local:
        print("extending arrays")
        #particle_active_local = np.resize(particle_active_local, new_length)
        particle_active_local = particle_active_local.resize(new_length)#, refcheck = False)
        # must set the added elements to False, NOT TRUE with .resize-method, missing/extra/new entries are filled with zero = false
        #particle_x_local = np.resize(particle_x_local, new_length)
        particle_x_local = particle_x_local.resize(new_length)#, refcheck = False)
        #particle_local = np.resize(particle_local, new_length)
        particle_local.resize(new_length)#, refcheck = False) # refcheck = True by default

# check if local arrays are bigger than needed (with a factor: shrink_if = 1/scaling_factor**3)
# old + new particles < shrink_if*old_size
# if they are, shrink them with a scaling_factor
if (active_n + received_n < shrink_if*particle_n_local):
    new_length = np.ceil(particle_n_local/scaling_factor)
    print("new_length:", new_length)
    #remove = np.floor((particle_n_local - particle_n_local)/scaling_factor)
    #print("remove number of elements:", remove)
    #print("TEST: it should equal:", particle_n_local - new_length)
        
    # if new length not equal old length
    # resize all local arrays
    if new_length != particle_n_local:
        print("shrinking arrays")
        #particle_active_local = np.resize(particle_active_local, new_length)
        particle_active_local.resize(new_length)
        #particle_x_local = np.resize(particle_x_local, new_length)
        particle_x_local.resize(new_length)
        #particle_local = np.resize(particle_local, new_length)
        particle_local.resize(new_length)#, refcheck = False) # refcheck = True by default

print("\nvalues after resizing:")
print("particle_n_local", particle_n_local)
print("particle_local", particle_local)
print("particle_x_local", particle_x_local)
print("old particle_active_local", particle_active_local)
print("active_n = number of old particles:", active_n)
print("number of incoming particles:", received_n)
print("length of local array:", particle_n_local)
print("\n")
        
# move all active particles to front of local arrays
move_active_to_front(particle_local, particle_x_local, particle_active_local, active_n)
        
## insert new particles
### TODO:
## should this be done for each row (ie. rank) in the arrays for communication?
## can this be done directly in the recv-for loop under # receiving?

# unpack (hstack) the list of arrays, (ravel/flatten can not be used for dtype=object)
#print("_flattend_ id:", np.hstack(recv_id_np))
#print("_flattend_ x:", np.hstack(recv_x_np))
#print("\n")

particle_local[active_n:active_n+received_n] = np.hstack(recv_id_np)
particle_x_local[active_n:active_n+received_n] = np.hstack(recv_x_np)
# set the received particles to active
particle_active_local[active_n:active_n+received_n]  = [True]*received_n

# set particle_n_local to the new size of the local arrays
particle_n_local = np.size(particle_local)
# this is not needed, the local arrays are transferred to the next iteration and the size etc. should be computed in the script

# TODO: se skype
#print("dtype", particle_local.dtype)
print("values after communication:")
print("particle_n_local", particle_n_local)
print("particle_local", particle_local)
print("particle_x_local", particle_x_local)
print("new particle_active_local", particle_active_local)

#comm.barrier()
#comm.wait()       
#print("sent", s)
#print("recv", r)
#print("fikk", f)
        
        #call MPI_IRECV(recv_id(irank+1)%p(1:Nrecv), Nrecv, MPI_DOUBLE, &irank, tag, MPI_COMM_WORLD, mpi_recv_request(tag, irank + 1), ierr)
        # tag = 2
        # mpi_recv_request_valid(tag, irank + 1) = .true.
        # call MPI_IRECV(recv_x(irank+1)%p(1:Nrecv), Nrecv, MPI_DOUBLE, &
            # irank, tag, MPI_COMM_WORLD, mpi_recv_request(tag, irank + 1), ierr)
        # tag = 3
        # mpi_recv_request_valid(tag, irank + 1) = .true.
        # call MPI_IRECV(recv_y(irank+1)%p(1:Nrecv), Nrecv, MPI_DOUBLE, &
            # irank, tag, MPI_COMM_WORLD, mpi_recv_request(tag, irank + 1), ierr)
        # tag = 4
        # mpi_recv_request_valid(tag, irank + 1) = .true.
        # call MPI_IRECV(recv_z(irank+1)%p(1:Nrecv), Nrecv, MPI_DOUBLE, &
            # irank, tag, MPI_COMM_WORLD, mpi_recv_request(tag, irank + 1), ierr)