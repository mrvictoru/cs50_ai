# helper function that save a list of figures into png files in the specified directory

import os
import matplotlib.pyplot as plt

def save_figures(figures, directory):
    # check if the directory exists
    if not os.path.exists(directory):
        # if not, create the directory
        os.makedirs(directory)
    
    for i, fig in enumerate(figures):
        fig.savefig(directory + 'figure_' + str(i) + '.png')
        plt.close(fig)