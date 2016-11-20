# script with functions to use in main
# this script handles plotting of particles

# Simen Mikkelsen, 2016-11-20

# plot.py
 
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('bmh')

##  INITIALISING start
figsize_x = 12
figsize_y = 6
# specify x- and y limits,
x_min = 0
x_max = 2
y_min = 0
y_max = 1

## INITIALISING end

## VARIABLES start


## VARIABLES end

# TODO: insert name
# XY is a two-component vector [x, y]
def plot(XY, time):
    fig = plt.figure(figsize = (figsize_x, figsize_y))
    #plt.scatter(x, y, lw = 0, marker = '.', s = 1)
    plt.scatter(XY[0,:], XY[1,:], marker = '.')# linewidth = 1, s = 1) # s = size
    # add text showing time, and set plot limits
    plt.text(1.65, 0.9, '$t = %s$' % time, size = 36)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    #plt.show()
    plt.savefig('plots\particles_t_%s.pdf' % time)