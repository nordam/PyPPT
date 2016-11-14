# script to generate binary python files with particle properties to be used in "communication_of_particles-py"
# this script generate data for 4 processes, distributed spatial evenly but "uneven on the ranks"

# Simen Mikkelsen, 2016-11-13

# example to run:
'''
python initialize_particles_4_ranks_uneven_distr.py
'''
import numpy as np
import os # for IO-paths

# initializing spatial properties
# numpy.random.rand gives random samples from a uniform distribution over [0, 1).
# thus the spatial properties are given by the distribution

## dividing of particles on ranks
# total number of particles
particle_n = 100
distr = np.array([0*particle_n, .4*particle_n, .7*particle_n, .9*particle_n, 1*particle_n], dtype='int')
# r0 = 400
# r1 = 700
# r2 = 900
# r2 = 1000

# IO
dir = os.path.dirname(os.path.realpath(__file__))
IO_path = os.path.join(dir, 'simulation', 'initialize')

# initializing of local particle array
for rank in range(4):

    particle_n_local = distr[rank+1] - distr[rank]
    particle_local = np.arange(distr[rank], distr[rank+1])

    particle_x_local = np.random.rand(particle_n_local)
    particle_y_local = np.random.rand(particle_n_local)
        
    # set all particles to active
    particle_active_local = np.ones(particle_n_local, dtype=bool)
    
    # save files
    file_path_id        = os.path.join(IO_path, 'particle_id' + str(rank))
    file_path_x         = os.path.join(IO_path, 'particle_x' + str(rank))
    file_path_y         = os.path.join(IO_path, 'particle_y' + str(rank))
    file_path_active    = os.path.join(IO_path, 'particle_active' + str(rank))
    
    np.save(file_path_id, particle_local)
    np.save(file_path_x, particle_x_local)
    np.save(file_path_y, particle_x_local)
    np.save(file_path_active, particle_active_local)
    
    print(particle_n_local)
    print(particle_local)
    print(particle_x_local)
    print(particle_y_local)
    print(particle_active_local)
