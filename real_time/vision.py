import cv2 as cv
import numpy as np


class Vision:

    # properties
    needle_img = None
    needle_w = 0
    needle_h = 0
    method = None

    # constructor
    def __init__(self, needle_img_path, method=cv.TM_CCOEFF_NORMED):
        # read img in
        self.needle_img = cv.imread(needle_img_path, cv.IMREAD_UNCHANGED)

        # save dimensions
        self.needle_w = self.needle_img.shape[1]
        self.needle_h = self.needle_img.shape[0]

        self.method = method

    def find(self, haystack_img, threshold=0.5, debug_mode=None):
        # adds greyscale to image
        # haystack_img = cv.cvtColor(haystack_img, cv.COLOR_BGR2GRAY)
        # needle_img = cv.cvtColor(needle_img, cv.COLOR_BGR2GRAY)

        # Template Matching
        result = cv.matchTemplate(haystack_img, self.needle_img, self.method)

        locations = np.where(result >= threshold)
        locations = list(zip(*locations[::-1]))

        # list of [x, y, w, h] rectangles
        rectangles = []
        for loc in locations:
            rect = [int(loc[0]), int(loc[1]), self.needle_w, self.needle_h]
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
        return points
