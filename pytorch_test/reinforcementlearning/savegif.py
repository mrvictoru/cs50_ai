# write a helper function that saves the frames as gif using matplotlib.animation

import matplotlib.animation as animation
import matplotlib.pyplot as plt

def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):
    # get a figure that has the same size as the frames
    plt.figure(figsize=(frames[0].shape[1]/72.0, frames[0].shape[0]/72.0), dpi=72)
    
    patch = plt.imshow(frames[0])
    plt.axis('off')
    
    def animate(i):
        patch.set_data(frames[i])
    
    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    
    anim.save(path + filename, writer='imagemagick', fps=60)