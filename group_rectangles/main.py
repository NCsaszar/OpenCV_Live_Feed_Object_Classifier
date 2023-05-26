import cv2 as cv
import numpy as np
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def findClickPositions(needle_img_path, haystack_img_path, threshold=0.7, debug_mode=None):
    haystack_img = cv.imread(haystack_img_path, cv.IMREAD_UNCHANGED)
    needle_img = cv.imread(needle_img_path, cv.IMREAD_UNCHANGED)
    # adds greyscale to image
    # haystack_img = cv.cvtColor(haystack_img, cv.COLOR_BGR2GRAY)
    # needle_img = cv.cvtColor(needle_img, cv.COLOR_BGR2GRAY)

    needle_w = needle_img.shape[1]
    needle_h = needle_img.shape[0]

    # Template Matching
    method = cv.TM_CCOEFF_NORMED
    result = cv.matchTemplate(haystack_img, needle_img, method)

    locations = np.where(result >= threshold)
    locations = list(zip(*locations[::-1]))

    # list of [x, y, w, h] rectangles
    rectangles = []
    for loc in locations:
        rect = [int(loc[0]), int(loc[1]), needle_w, needle_h]
        rectangles.append(rect)
        rectangles.append(rect)

    rectangles, weights = cv.groupRectangles(rectangles, 1, 0.3)
    points = []
    if len(rectangles):
        print('Found needle.')

        line_color = (0, 255, 0)
        line_type = cv.LINE_4
        thick = thickness = 2
        marker_color = (0, 0, 255)
        marker_type = cv.MARKER_CROSS

        for (x, y, w, h) in rectangles:
            # Determine the center position
            center_x = x + int(w/2)
            center_y = y + int(h/2)
            # save the points
            points.append((center_x, center_y))

            if debug_mode == 'rectangles':
                top_left = (x, y)
                bottom_right = (x + w, y + h)
                # draw box
                cv.rectangle(haystack_img, top_left,
                             bottom_right, line_color, thick, line_type)
            elif debug_mode == 'points':
                cv.drawMarker(haystack_img, (center_x, center_y),
                              marker_color, marker_type, 100)
        if debug_mode:
            cv.imshow("matches", haystack_img)
            cv.waitKey()
    return points


points = findClickPositions(
    '../images/testneedle2.jpg', '../images/testhay.jpg', threshold=0.4, debug_mode='rectangles')
print(points)
