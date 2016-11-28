# script with functions to use in main
# this script handles plotting of particles

# Simen Mikkelsen, 2016-11-20

# plot.py

import os
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns # plot style

import IO

plt.style.use('bmh')

##  INITIALISING start
figsize_x = 12
figsize_y = 6
# specify x- and y limits,
x_min = 0
x_max = 2
y_min = 0
y_max = 1

# file extension
#file_extension = '.pdf'
file_extension = '.png'

## INITIALISING end

## VARIABLES start


## VARIABLES end

## FUNCTIONS start

# XY is a two-component vector [x, y]
def plot(rank, XY, time, dt, name = ''):
    fig = plt.figure(figsize = (figsize_x, figsize_y))
    #plt.scatter(x, y, lw = 0, marker = '.', s = 1)
    plt.scatter(XY[0,:], XY[1,:], marker = '.', s = 5)#, linewidth = 1)# s = size
    # add text showing time, and set plot limits
    plt.text(1.65, 0.9, 'rank = %s\n$t = %s$' % (rank, time), size = 36)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    #plt.show()
    plt.savefig(os.path.join('plots', 'particles_time%s%s_rank%s_dt%s%s' % (time, name, rank, dt, file_extension)))

def plot_from_files(time, mpi_size, dt, input = False):
    fig = plt.figure(figsize = (figsize_x, figsize_y))
    colors = 'brcykgmk'
    for rank in range (mpi_size):
        # load particles
        id, active, XY = IO.load_grid_of_particles(rank, time, input)
        plt.scatter(XY[0,active], XY[1,active], marker = '.', s = 6, label = 'rank = %s' % rank, color = colors[rank])#label = 'rank = %s' % rank)#, linewidth = 1)# s = size
        
        #plt.text(iter(['rank 0', 'rank 1', 'rank 2', 'rank 3']))
        #plt.text(1.65, 0.9, 'rank = %s\n$t = %s$' % (rank, time), size = 36)
        
        # print('rank', rank)
        # print('x', XY[0,active])
        # print('\n')
        # print('y', XY[1,active])
        
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.legend(scatterpoints = 1)
    #plt.show()
    plt.savefig(os.path.join('plots', 'allranks_particles_time%s_dt%s%s' % (time, dt, file_extension)))
    
    
    
## FUNCTIONS end

if __name__ == '__main__':
    
    t = 5.0
    total_ranks = 4
    
    plot_from_files(t, total_ranks, dt = '-')
    # todo fargelegg hver rank
    