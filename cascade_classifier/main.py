import os
import numpy as np
from time import time
import cv2 as cv
from windowcapture import WindowCapture
from vision import Vision

os.chdir(os.path.dirname(os.path.abspath(__file__)))

wincap = WindowCapture('Tarn')
# WindowCapture.list_window_names()

# load trained model
cascade_boss = cv.CascadeClassifier('cascade/cascade.xml')

# load an empty vision class
vision_boss = Vision(None)

loop_time = time()
while (True):
    # get an updated image of the window
    screenshot = wincap.get_screenshot()

    # do object detection
    rectangles = cascade_boss.detectMultiScale(screenshot)

    # draw the detection results on the original image
    detection_image = vision_boss.draw_rectangles(screenshot, rectangles)

    # display the images
    cv.imshow('Matches', detection_image)

    # print('FPS {}'.format(1 / (time() - loop_time)))
    loop_time = time()

    # press 'q' with the output window focused to exit
    # wait 1 ms every loop to process key presses
    key = cv.waitKey(1)
    if key == ord('q'):
        cv.destroyAllWindows()
        break
    elif key == ord('f'):
        cv.imwrite('positive/{}.jpg'.format(loop_time), screenshot)
    elif key == ord('d'):
        cv.imwrite('negative/{}.jpg'.format(loop_time), screenshot)

print('Done')
