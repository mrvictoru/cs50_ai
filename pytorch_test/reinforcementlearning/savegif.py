# write a helper function that saves the frames as gif using matplotlib.animation

import matplotlib.animation as animation
import matplotlib.pyplot as plt

def save_frames_as_gif(frames, filename):
    fig = plt.figure()

    # create an animation object
    ani = animation.ArtistAnimation(fig, frames, interval=40, blit=True, repeat_delay=2500)

    ani.save(filename, writer='pillow')