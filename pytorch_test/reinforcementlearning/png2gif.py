# animate png into gif

import imageio
import os

def save_gif(directory, gif_name):
    # get the list of file names
    filenames = os.listdir(directory)
    # exclude the non-png files
    filenames = [filename for filename in filenames if filename.endswith('.png')]
    # sort the file names
    filenames = sorted(filenames)
    # create the list of file paths
    filenames = [directory + '/' + filename for filename in filenames]
    # create the list of images
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    # save the gif
    imageio.mimsave(gif_name, images, duration=0.2)

# create main function
def main():
    # ask for user input, the directory containing the png files, and the name for the gif file
    directory = input("Please enter the directory containing the png files: ")
    gif_name = input("Please enter the name of the gif file to be created (e.g. animate.gif): ")
    # call the save_gif function to create the gif
    save_gif(directory, gif_name)
    # print success message
    print("Success!")

# call main function
if __name__ == "__main__":
    main()
