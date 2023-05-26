import cv2 as cv
import numpy as np
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))


haystack_img = cv.imread('../images/testhay.jpg', cv.IMREAD_UNCHANGED)
needle_img = cv.imread('../images/testneedle.jpg', cv.IMREAD_UNCHANGED)
# adds greyscale to image
# haystack_img = cv.cvtColor(haystack_img, cv.COLOR_BGR2GRAY)
# needle_img = cv.cvtColor(needle_img, cv.COLOR_BGR2GRAY)

result = cv.matchTemplate(haystack_img, needle_img, cv.TM_CCOEFF_NORMED)

threshold = 0.9

locations = np.where(result >= threshold)

locations = list(zip(*locations[::-1]))
print(locations)

if locations:
    print('Found needle.')

    needle_w = needle_img.shape[1]
    needle_h = needle_img.shape[0]
    line_color = (0, 255, 0)
    line_type = cv.LINE_4
    thick = thickness = 2

    for loc in locations:
        top_left = loc
        bottom_right = (top_left[0] + needle_w, top_left[1] + needle_h)
        # draw box
        cv.rectangle(haystack_img, top_left,
                     bottom_right, line_color, thick, line_type)
        cv.imshow("matches", haystack_img)
        cv.waitKey()

    else:
        print('Needle not found.')
