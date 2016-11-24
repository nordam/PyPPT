# script with functions to use in main-file
# this script is used for debugging

# Simen Mikkelsen, 2016-11-21

# debugging.py

import numpy as np

## FUNCTIONS start
def print_active_vs_XY_shape(XY, active, header = None):
    if header:
        print(header)
    print('shape active:', active.shape)
    print('sum active:', np.sum(active))
    print('shape XY:', XY.shape)
    print('\n')
    
    #print('active:', active)
    #print('XY:', XY)