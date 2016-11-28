# script with functions to use in main
# this script handle the 'saving of file'-part

# Simen Mikkelsen, 2016-11-19
# save.py

import numpy as np
import os # for IO-paths

##  INITIALISING start
file_extension = '.npy'
file_folder = 'files'
input_folder = 'input'
output_folder = 'output'

particle_x_name = 'particle_x'
particle_y_name = 'particle_y'
id_name = 'particle_id'
active_name = 'particle_active'

## INITIALISING end

## VARIABLES start
cdir = os.path.dirname(os.path.realpath(__file__))
IO_path_input = os.path.join(cdir,file_folder, input_folder)
IO_path_output = os.path.join(cdir,file_folder, output_folder)

## VARIABLES end

## FUNCTIONS start
## FUNCTIONS end
def save_array_binary_file(array, name, time, rank, input = False):
    if input:
        file_path = os.path.join(IO_path_input, 'time%s_%s_rank%s' % (time, name, rank))
    else:
        file_path = os.path.join(IO_path_output, 'time%s_%s_rank%s' % (time, name, rank))
    np.save(file_path, array)

def load_array_binary_file(name, time, rank, input = False):
    if input:
        file_path = os.path.join(IO_path_input, 'time%s_%s_rank%s%s' % (time, name, rank, file_extension))
    else:
        file_path = os.path.join(IO_path_output, 'time%s_%s_rank%s%s' % (time, name, rank, file_extension))
    return np.load(file_path)
    
def create_grid_of_particles(N, w):
    # create a grid of N evenly spaced particles
    # N is rounded down to nearest perfect square
    N = (int(np.sqrt(N)))**2
    print('Creating grid of particles')
    print('N: %s, w: %s' % (N, w))
    # covering a square patch of width and height w
    # centered on the region 0 < x < 2, 0 < y < 1
    ids = np.arange(N)
    active = np.ones(N, dtype=bool)
    x  = np.linspace(1.0-w/2, 1.0+w/2, int(np.sqrt(N)))
    y  = np.linspace(0.5-w/2, 0.5+w/2, int(np.sqrt(N)))
    x, y = np.meshgrid(x, y)
    return ids, active, np.array([np.ravel(x), np.ravel(y)])

def save_grid_of_particles(ids, active, XY, time, rank, input = False):
    # XY is a two-component vector [x, y]
    save_array_binary_file(ids,     id_name,         time, rank, input)
    save_array_binary_file(active,  active_name,     time, rank, input)
    save_array_binary_file(XY[0,:], particle_x_name, time, rank, input)
    save_array_binary_file(XY[1,:], particle_y_name, time, rank, input)

def save_empty_grid(time, rank, input = False):
    save_array_binary_file(np.ndarray(0, dtype=int ),   id_name,         time, rank, input)
    save_array_binary_file(np.ndarray(0, dtype=bool),   active_name,     time, rank, input)
    save_array_binary_file(np.ndarray(0),               particle_x_name, time, rank, input)
    save_array_binary_file(np.ndarray(0),               particle_y_name, time, rank, input)

def load_grid_of_particles(rank, time, input = False):
    # XY is a two-component vector [x, y]
    if input:
        file_path_id = os.path.join(IO_path_input, 'time%s_%s_rank%s%s' % (time, id_name, rank, file_extension))
        file_path_active = os.path.join(IO_path_input, 'time%s_%s_rank%s%s' % (time, active_name, rank, file_extension))
        file_path_x = os.path.join(IO_path_input, 'time%s_%s_rank%s%s' % (time, particle_x_name, rank, file_extension))
        file_path_y = os.path.join(IO_path_input, 'time%s_%s_rank%s%s' % (time, particle_y_name, rank, file_extension))
    else:
        file_path_id = os.path.join(IO_path_output, 'time%s_%s_rank%s%s' % (time, id_name, rank, file_extension))
        file_path_active = os.path.join(IO_path_output, 'time%s_%s_rank%s%s' % (time, active_name, rank, file_extension))
        file_path_x = os.path.join(IO_path_output, 'time%s_%s_rank%s%s' % (time, particle_x_name, rank, file_extension))
        file_path_y = os.path.join(IO_path_output, 'time%s_%s_rank%s%s' % (time, particle_y_name, rank, file_extension))    
    return (np.load(file_path_id),
            np.load(file_path_active),
            np.array([np.load(file_path_x), np.load(file_path_y)])
            )

## FUNCTIONS end

if __name__ == '__main__':
    N = 1000
    w = 0.1
    t = 0
    total_ranks = 4
    i, a, xy = create_grid_of_particles(N, w)
    save_grid_of_particles(i, a, xy, t, rank=0, input=True)
    # save empty grids for the other ranks, in this case for 4 total ranks
    
    for i in range(1, total_ranks):
        print('creating empty arrays for rank:', i)
        save_empty_grid(t, i, input = True)