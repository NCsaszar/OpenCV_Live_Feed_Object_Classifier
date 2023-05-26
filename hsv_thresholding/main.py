import os
import numpy as np
from time import time
import cv2 as cv
from windowcapture import WindowCapture
from vision import Vision

os.chdir(os.path.dirname(os.path.abspath(__file__)))

wincap = WindowCapture('Tarn')
# WindowCapture.list_window_names()
vision_boss = Vision('../images/bossmaster.jpg')

# initialize the trackbar window
vision_boss.init_control_gui()

loop_time = time()
while (True):
    screenshot = wincap.get_screenshot()

    # pre processing of the image with hsv
    output_image = vision_boss.apply_hsv_filter(screenshot)

    # # object detection
    # rectangles = vision_boss.find(screenshot, 0.3)

    # # output image
    # output_image = vision_boss.draw_rectangles(screenshot, rectangles)

    # draw rects
    cv.imshow('Matches', output_image)

    # print('FPS {}'.format(1 / (time() - loop_time)))
    loop_time = time()

    # press 'q' with the output window focused to exit
    # wait 1 ms every loop to process key presses
    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        break

print('Done')
