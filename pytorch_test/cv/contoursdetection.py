# this function return img with contours and img with bounding rectangle and area of the contour on the original image

import cv2
import numpy


def getContours(ogimg , imgDilate, areaThreshold):
    imgContour = ogimg.copy()
    imgDetect = ogimg.copy()
    # find contours in the image
    contours, hierarchy = cv2.findContours(imgDilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # loop through all the contours
    for cnt in contours:
        # get area of contour
        area = cv2.contourArea(cnt)
        # check if area is greater than certain value
        if area > areaThreshold:
            # draw contour on the original image
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 3)

            # get the boundary length for the contour and count the number of side of the contour using approxPolyDP
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)

            # create a bounding rectangle around the contour
            x, y, w, h =  cv2.boundingRect(approx)
            cv2.rectangle(imgDetect, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # put the amount of points in the contour on the original image
            cv2.putText(imgDetect, "Points: " + str(len(approx)), (x + w +20, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            # put the area of the contour on the original image
            cv2.putText(imgDetect, "Area: " + str(int(area)), (x + w +20, y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    return imgContour, imgDetect

