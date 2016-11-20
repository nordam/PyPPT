# script to generate binary python files with particle properties to be used in "transport_and_communication_of_particles.py"
# this script generate data for 4 processes
# distributed spatial evenly as a rectangle
# evenly distributed on ranks

# Simen Mikkelsen, 2016-11-15

# example to run:
'''
python initialize_particle_rectangle.py
'''
import numpy as np
import os # for IO-paths
import matplotlib.pyplot as plt
plt.style.use('bmh')

# number of ranks
mpi_size = 1

# specify x- and y limits,
x_min = 0
y_min = 0
x_max = 2 # x_max
y_max = 1 # y_max

# rectangular grid of points
# specify resolution
Nx = 5    
Ny = 5
# specify limit of particle rectangle
particles_x_min = 0.9
particles_x_max = 1.1
particles_y_min = 0.4
particles_y_max = 0.6

# create lists of all x and y positions
X  = np.linspace(particles_x_min, particles_x_max, Nx, endpoint = False)
Y  = np.linspace(particles_y_min, particles_y_max, Ny, endpoint = False)

# Use meshgrid to turn lists into rank 2 arrays
# of x and y positions
XY = np.meshgrid(X, Y)

particle_n = Nx*Ny
particle_id = np.arange(particle_n)
particle_x = np.hstack(XY[:][0])
particle_y = np.hstack(XY[:][1])


# IO
dir = os.path.dirname(os.path.realpath(__file__))
IO_path = os.path.join(dir, 'simulation', 'rectangle')

# distribution of particles

# divide particles evenly on ranks
for rank in range(mpi_size):

    
    particle_local = particle_id[rank::mpi_size]
    particle_n_local = np.size(particle_local)
    
    particle_x_local = particle_x[rank::mpi_size]
    particle_y_local = particle_y[rank::mpi_size]
        
    # set all particles to active
    particle_active_local = np.ones(particle_n_local, dtype=bool)
    
    # save files
    file_path_id        = os.path.join(IO_path, 'particle_id' + str(rank))
    file_path_x         = os.path.join(IO_path, 'particle_x' + str(rank))
    file_path_y         = os.path.join(IO_path, 'particle_y' + str(rank))
    file_path_active    = os.path.join(IO_path, 'particle_active' + str(rank))
    
    np.save(file_path_id, particle_local)
    np.save(file_path_x, particle_x_local)
    np.save(file_path_y, particle_y_local)
    np.save(file_path_active, particle_active_local)
    
    print("particle_n_local:", particle_n_local)
    print("particle_local:", particle_local)
    print(particle_x_local)
    print(particle_y_local)
    print(particle_active_local*1)
    
    # plot test for each rank
    fig = plt.figure(figsize = (12,6))
    plt.scatter(particle_x_local, particle_y_local, marker = '.', c = '#348ABD')
    # add text showing time, and set plot limits
    plt.text(1.65, 0.9, '$t = %s$' % 0, size = 36)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

plt.show()
