import cv2 as cv
import numpy as np

haystack_img = cv.imread("testhay.jpg", cv.IMREAD_UNCHANGED)
needle_img = cv.imread("testneedle.jpg", cv.IMREAD_UNCHANGED)
# adds greyscale to image
# haystack_img = cv.cvtColor(haystack_img, cv.COLOR_BGR2GRAY)
# needle_img = cv.cvtColor(needle_img, cv.COLOR_BGR2GRAY)

result = cv.matchTemplate(haystack_img, needle_img, cv.TM_CCOEFF_NORMED)


# get best match position
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

print('Best match top left pos: %s' % str(max_loc))
print('Best match confidence: %s' % max_val)

threshold = 0.8
if max_val >= threshold:
    print('Found needle.')

    # dimensions of needle image
    needle_w = needle_img.shape[1]
    needle_h = needle_img.shape[0]

    top_left = max_loc
    bottom_right = (top_left[0] + needle_w, top_left[1] + needle_h)

    cv.rectangle(haystack_img, top_left, bottom_right, color=(
        0, 255, 0), thickness=2, lineType=cv.LINE_4)
    cv.imshow("Result", haystack_img)
    cv.waitKey()
else:
    print('Needle not found.')
