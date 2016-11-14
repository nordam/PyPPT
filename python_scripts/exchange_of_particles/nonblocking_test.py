
import numpy as np
import os # for IO-paths

from mpi4py import MPI
from matplotlib import pyplot as plt
#plt.style.use('bmh')

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
mpi_size = comm.Get_size()


# Create some random numbers
N = 500000
x = np.random.random(size = N).astype(np.float64)

# Create arrays to receive data into
R = np.zeros((mpi_size, N))

# Copy own data to correct row
R[rank, :] = x

# send to all other ranks:
send_requests = [None] * mpi_size
for i in range(mpi_size):
    if i != rank:
        send_requests[i] = comm.isend(R[rank, :], dest = i)

# Set up non-blocking receives
recv_requests = [None] * mpi_size
for i in range(mpi_size):
    if i != rank:
        # This part is crucial. irecv requires a temporary buffer
        # (you can call it with buf=None, but that will fail for large
        # messages). The returned request object will hold a reference
        # to the buffer, so the variable name can be re-used.
        # Note that an overhead is required, as some extra stuff
        # is also sent. 1000 doubles equals 8 kilobytes.
        buf = np.zeros(N + 1000, dtype = np.float64)
        recv_requests[i] = comm.irecv(buf = buf, source = i)

# Obtain data from completed requests
# only at this step is the data actually returned.
for i in range(mpi_size):
    if i != rank:
        R[i,:] = recv_requests[i].wait()

# Make sure this rank does not exit until sends have completed
for i in range(mpi_size):
    if i != rank:
        send_requests[i].wait()

# Confirm that everything went smoothly by printing
# the last number from each row of R
# each rank should have the same data
print('rank: ', rank, R[:,-1])

