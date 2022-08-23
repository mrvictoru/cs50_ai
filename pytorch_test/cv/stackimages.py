# functions that stack images
# script built from Murtaza's workshop
import cv2
import numpy as np

# the function takes the scale values and the array of image and stack accordingly
def stackImages(scale,imgArray):
    # get the number the rows and columns of stacking from the image array
    rows = len(imgArray)
    cols = len(imgArray[0])
    # get the width and height of the image
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    # check if there is more than one row
    rowsAvailable = isinstance(imgArray[0], list)
    #
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                # check if the next image's dimension match the first image
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    # if so scale the image by scale
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    #if not resize the image to the first image's dimension
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                # check image color space, if it is not RGB, convert it to BGR
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
                if imgArray[x][y].shape[2] == 4: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_BGRA2BGR)
        # create a blank image of the same dimension as the first image        
        imageBlank = np.zeros((height, width, 3), np.uint8)

        # create dummy array to store the stack horizontal images
        hor = [imageBlank]*rows
        
        # stack the images horizontally
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])

        # stack the images vertically
        ver = np.vstack(hor)
    
    # if not then just stack the image horizontally
    else:
        for x in range(0, rows):
            # check if the next image's dimension match the first image
            if imgArray[x].shape[:2] == imgArray[0].shape [:2]:
                # if so scale the image by scale
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                #if not resize the image to the first image's dimension
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            # check image color space, if it is not RGB, convert it to BGR
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
            if imgArray[x].shape[2] == 4: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_BGRA2BGR)
        # stack the images horizontally
        hor = np.hstack(imgArray)
        ver = hor
    return ver