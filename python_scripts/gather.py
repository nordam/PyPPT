import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

local_array = np.zeros((2,2))
local_array[:] = rank

if rank == 0:
    global_array = np.zeros((4,4))
else:
    global_array = None

comm.Gather(local_array, global_array, root = 0)

print(rank, global_array)
