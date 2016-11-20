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

## INITIALISING end

## VARIABLES start
dir = os.path.dirname(os.path.realpath(__file__))
IO_path_input = os.path.join(dir,file_folder, input_folder)
IO_path_output = os.path.join(dir,file_folder, output_folder)

## VARIABLES end
    
def save_array_binary_file(array, name, time, rank):
    #print(array)
    file_path = os.path.join(IO_path_output, 'time_%s_%s_rank_%s' % (time, name, rank))
    np.save(file_path, array)
    
def load_array_binary_file(name, time, rank):
    file_path = os.path.join(IO_path_input, 'time_%s_%s_rank_%s%s' % (time, name, rank, file_extension))
    #print('input file path', file_path)
    return np.load(file_path)
    
def create_grid_of_particles(N, w):
    # create a grid of N evenly spaced particles
    # covering a square patch of width and height w
    # centered on the region 0 < x < 2, 0 < y < 1
    id = np.arange(N)
    x  = np.linspace(1.0-w/2, 1.0+w/2, int(np.sqrt(N)))
    y  = np.linspace(0.5-w/2, 0.5+w/2, int(np.sqrt(N)))
    x, y = np.meshgrid(x, y)
    return id, np.array([np.ravel(x), np.ravel(y)])
    
def save_grid_of_particles(id, XY, time, rank):
    # XY is a two-component vector [x, y]
    #save_array_binary_file(id,      id_name,         time, rank)
    save_array_binary_file(XY[0,:], particle_x_name, time, rank)
    save_array_binary_file(XY[1,:], particle_y_name, time, rank)