# example to run:
#mpiexec -n 2 python ***.py
# where 1 in number of processes

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

sendbuf = np.zeros((10, 10), dtype='i') + rank
print("size:", size, "rank", rank, "data send : \n", sendbuf)
recvbuf = None
if rank == 0:
    recvbuf = np.zeros([size, 10, 10], dtype='i')
comm.Gather(sendbuf+1, recvbuf, root=0)
if rank == 0:
    print("rank 0 data: \n", recvbuf)
    #for i in range(size):
    #    assert np.allclose(recvbuf[i,:], i)