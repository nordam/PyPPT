# script with functions to use in main-file
# this script handle the "communication of particles between ranks"-part

# Simen Mikkelsen, 2016-11-21

# communication.py
'''
def exchange(X):
    # Handle all the communication stuff here
    # return the updated particle arrays
    # (which may be of a different length now)
    return X
'''
### TODO:
#implement 'find biggest factor'-function or another dynamical function to determine number of cells in each direction

import numpy as np
### import mpi4py here?

##  INITIALISING start
# number of cells in each direction (only divide x-direction initially)
cell_x_n = 10
cell_y_n = 0
cell_n = cell_x_n + cell_y_n

# scaling factor when expanding/scrinking local arrays
scaling_factor = 1.25 ## variable
shrink_if = 1/(scaling_factor**3)

# the particles are defined with its properties in several arrays
# one tag for each properties which are communicated to other ranks
# tags: id, x-pos, y-pos
# other properties: active-status
tag_n = 3

# buffer overhead to use in memory reservation for non-blocking communication
buffer_overhead = 1000

## INITIALISING end

## VARIABLES start

## VARIABLES end

## secondary FUNCTIONS start

# function to find the corresponding rank of a cell
# this function determines how the cells are distributed to ranks
### TODO: discussion: how to do this distribution
def find_rank_from_cell(cell_id):
    return (cell_id % mpi_size)
    
# function to find the corresponding cell of a position
# this function determines how the cells are distributed geometrically
### TODO: discussion: how to do this distribution
def find_cell_from_position(x, y):
    return (np.floor(((x - x_start)/(x_len))*(cell_x_n))) # for 1D

# send_n_array: array to show how many particles should be sent from one rank to the others
# filled out locally in each rank, then communicated to all other ranks
# rows represent particles sent FROM rank = row number (0 indexing)
# column represent particles sent TO rank = row number (0 indexing)
# function to fill out the array showing number of particles need to be sent from a given rank given thelocal particles there
# local particles are the particles who belonged to the rank before the transport of particles.
# some of the particles may have to been moved to a new rank if they have been moved to a cell belonging to a new rank
# send_to: array to show which rank a local particle needs to be sent to. or -1 if it should stay in the same rank
def global_communication_array(mpi_size, rank, particle_n, particle_x, particle_y, particle_active):
    # reset arrays telling which particles are to be sent
    send_to = np.zeros(particle_n, dtype=int) # local
    send_to[:] = -1
    send_n_array = np.zeros((mpi_size, mpi_size), dtype=int)
    for i in range(particle_n):
        # only check if the particle is active
        if particle_active[i]:
            # find the rank of the cell of which the particle (its position) belongs to
            particle_rank = find_rank_from_cell(find_cell_from_position(particle_x[i], particle_y[i]))
            # if the particle's new rank does not equal the current rank (for the given process), it should be moved
            if particle_rank != rank:
                send_n_array[int(rank)][int(particle_rank)] = send_n_array[int(rank)][int(particle_rank)] + 1
                send_to[i] = particle_rank
                # converted indices to int to not get 'deprecation warning'
    return send_to, send_n_array

# function to reallocate active particles to the front of the local arrays
# active_n = number of active particles after deactivation of particles sent to another rank, but before receiving.
# aka. particles that stays in its own rank
def move_active_to_front(particle_id, particle_x, particle_y, particle_active, active_n):
    particle_id[:active_n] = particle_id[particle_active]
    particle_x[:active_n] = particle_x[particle_active]
    particle_y[:active_n] = particle_y[particle_active]
    # set the corresponding first particles to active, the rest to false
    particle_active[:active_n] = True
    particle_active[active_n:] = False
    return particle_id, particle_x, particle_y, particle_active    

## secondary FUNCTIONS end

## main FUNCTION start

# all variables taken in by exchange() are local variables for the given rank (except mpi_size)
def exchange(mpi_size,
            rank,
            particle_id,
            particle_x,
            particle_y,
            particle_active):
    
    # compute "global communication array"
    # with all-to-all communication
    
    # length of local particle arrays
    # note: not necessary equal to number of active particles
    particle_n = size(particle_id)
    send_to, send_n = global_communication_array(mpi_size, rank, particle_n, particle_x, particle_y, particle_active)
    
    # all nodes receives results with a collective 'Allreduce'
    
    # mpi4py requires that we pass numpy objects (byte-like objects)
    send_n_global = np.zeros((mpi_size, mpi_size), dtype=int)
    comm.Allreduce(send_n_local , send_n_global , op=MPI.SUM)
    
    # each rank communicate with other ranks if it sends or receives particles from that rank
    # this information is now given in the "global communication array"

    # point-to-point communication of particles
        
    # using list of arrays for communication of particle properties
    # initializing "communication arrays": send_*** and recv_***
    # send_**: list of arrays to hold particles that are to be sent from a given rank to other ranks,
    #   where row number corresponds to the rank the the particles are send to
    # recv_**: list of arrays to hold particles that are to be received from to a given rank from other ranks,
    #   where row number corresponds to the rank the particles are sent from
    
    send_id = []
    send_x = []
    send_y = []
    recv_id = []
    recv_x = []
    recv_y = []
    
    # total number of received particles
    received_n = np.sum(send_n_global, axis = 0)[rank]
    
    for irank in range(mpi_size):
    
        # find number of particles to be received from irank (sent to current rank)
        Nrecv = send_n_global[irank, rank]
        # append recv_id with the corresponding number of elements
        recv_id.append([None]*Nrecv)
        recv_x.append([None]*Nrecv)
        recv_y.append([None]*Nrecv)
                
        # find number of particles to be sent to irank (from current rank)
        Nsend = send_n_global[rank, irank]
        # append send_id with the corresponding number of elements
        send_id.append([None]*Nsend)
        send_x.append([None]*Nsend)
        send_y.append([None]*Nsend)

    # counter to get position in send_** for a particle to be sent
    send_count = np.zeros(mpi_size, dtype=int)
    
    # iterate over all local particles to allocate them to send_** if they belong in another rank
    for i in range(particle_n):
        # if particle is active (still a local particle) and should be sent to a rank (-1 means that the particle already is in the correct rank)
        if (particle_active[i] and send_to[i] != -1):
            # fill the temporary communication arrays (send_**) with particle and it's properties
            send_id[send_to[i]][send_count[send_to[i]]] = i
            send_x[send_to[i]][send_count[send_to[i]]] = particle_x[i]
            send_y[send_to[i]][send_count[send_to[i]]] = particle_y[i]
            
            # deactivate sent particle
            particle_active[i] = False
            # increment counter to update position in temporary communication arrays (send_**)
            send_count[send_to[i]] = send_count[send_to[i]] + 1

    # actual exchange of particle properties follows
    
    # must convert the list of arrays which are to be communicated to numpy objects (byte-like objects)
    # this is not done before because np.ndarrays does not support a "list of arrays" if the arrays does not have equal dimensions
    send_id_np = np.array(send_id)
    recv_id_np = np.array(recv_id)
    send_x_np = np.array(send_x)
    recv_x_np = np.array(recv_x)
    send_y_np = np.array(send_y)
    recv_y_np = np.array(recv_y)
    
    # requests to be used for non-blocking send and receives
    send_request = [None]*mpi_size
    recv_request = [None]*mpi_size
    
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
                send_request[irank] = comm.isend(send_id_np[irank][0:Nsend], dest = irank, tag = 1) # could have written [:] insted of [0:Nsend]
                comm.send(send_x_np[irank][0:Nsend], dest = irank, tag = 2)
                comm.send(send_y_np[irank][0:Nsend], dest = irank, tag = 3)
                #print('sending:', send_id_np[irank][0:Nsend])#, send_x_np[irank][0:Nsend])
    print('send_id_np:', send_id_np)
    
    # receiving

    for irank in range(mpi_size):
        if (irank != rank):
            # number of particles irank sends to rank (number of particles rank recieves from irank)
            Nrecv = send_n_global[irank, rank]
            # only receive if there is something to recieve
            if (Nrecv > 0):
                #print('rank:', rank, 'receiving', Nrecv, 'particles from', irank)
                # use tags to separate communication of different arrays/properties
                # tag uses 1-indexing so there will be no confusion with the default tag = 0
                buf = np.zeros(Nrecv+buffer_overhead, dtype = np.int)
                #print("buf:", buf)
                recv_request[irank] = comm.irecv(buf = buf, source = irank, tag = 1)
                
                #recv_id_np[irank][0:Nrecv] = comm.recv(buf=None, source = irank, tag = 1)
                recv_x_np[irank][0:Nrecv] = comm.recv(buf=None, source = irank, tag = 2)
                recv_y_np[irank][0:Nrecv] = comm.recv(buf=None, source = irank, tag = 3)
                #print('receiving:', recv_id_np[irank][0:Nrecv])#, send_x_np[irank][0:Nsend])
    
    # obtain data from completed requests
    # only at this step is the data actually returned.
    for irank in range(mpi_size):
        if irank != rank:
            recv_id_np[irank][:] = recv_request[irank].wait()
            
    print('recv_id_np:', recv_id_np)#,',' , recv_x_np)
    
    # make sure this rank does not exit until sends have completed
    for irank in range(mpi_size):
        if irank != rank:
            send_request[irank].wait()

    # number of active particles after sending
    active_n = np.sum(particle_active)

    # move all active particles to front of local arrays
    if (active_n > 0):
        particle_id, particle_x, particle_y, particle_active = move_active_to_front(particle_id, particle_x, particle_y, particle_active, active_n)

    # resize local arrays if needed
    
    # current scaling factor = 1.25
    ### TODO: add ceil/floor directly in if-check?
    
    # check if local arrays have enough free space, if not, allocate a 'scaling_factor' more than needed
    if (active_n + received_n > particle_n):
        new_length = int(np.ceil((active_n + received_n)*scaling_factor))
        # if new length is not equal old length: resize all local arrays
        if new_length != particle_n:
            #print('extending arrays to new length:', new_length)
            # with .resize-method, missing/extra/new entries are filled with zero (false in particle_active)
            particle_active.resize(new_length)#, refcheck = False)
            particle_id.resize(new_length)#, refcheck = False) # refcheck = True by default
            particle_x.resize(new_length)
            particle_y.resize(new_length)
            
    # check if local arrays are bigger than needed (with a factor: shrink_if = 1/scaling_factor**3)
    # old + new particles < shrink_if*old_size
    # if they are, shrink them with a scaling_factor
    if (active_n + received_n < shrink_if*particle_n):
        new_length = int(np.ceil(particle_n/scaling_factor))
        # if new length is not equal old length: resize all local arrays
        if new_length != particle_n:
            #print('shrinking arrays to new length:', new_length)
            particle_active.resize(new_length)
            particle_id.resize(new_length)#, refcheck = False) # refcheck = True by default
            particle_x.resize(new_length)
            particle_y.resize(new_length)

    # add the received particles to local arrays     

    # unpack (hstack) the list of arrays, (ravel/flatten can not be used for dtype=object)

    print("received_n:", received_n)
    print("active_n:", active_n)

    if received_n > 0:
        particle_id[active_n:active_n+received_n] = np.hstack(recv_id_np)
        particle_x[active_n:active_n+received_n] = np.hstack(recv_x_np)
        particle_y[active_n:active_n+received_n] = np.hstack(recv_y_np)
    # set the received particles to active
        particle_active[active_n:active_n+received_n]  = [True]*received_n
    
    # optional printing for debugging
    # print the new values
        print('\nnew length of local arrays:', particle_n)
        print("new local particles:", particle_id)
        print("new active particles:", particle_active*1) # *1 to turn the output into 0 and 1 instead of False and True
    else:
        print("\nno received particles/aka no changes")

    # print global array
    if rank == 0:
        print('\nglobal_array:\n', send_n_global)
    
    # return the updated particle arrays
    return (particle_id,
            particle_x,
            particle_y,
            particle_active)