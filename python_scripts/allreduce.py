import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


assert size == 4

local_array = np.zeros((4,4))

if rank == 0:
    local_array[0:2, 0:2] = rank + 1
if rank == 1:
    local_array[2:4, 0:2] = rank + 1
if rank == 2:
    local_array[0:2, 2:4] = rank + 1
if rank == 3:
    local_array[2:4, 2:4] = rank + 1

global_array = np.zeros((4,4))

comm.Allreduce(local_array, global_array, op = MPI.SUM)

print(rank, global_array)
