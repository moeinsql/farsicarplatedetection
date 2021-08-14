import croplicenseplate as clp
import checkdetectedplate as chkplt
import detectchars as dc
import cv2 as cv
import sys

# parameters in detectchars
MIN_HW_RATIO = 0.2
MAX_DISTANCE_SIZE = 10000000.0
DISTANCE_SIZE = 2000000.0
DISTANCE_STEP = 2000000.0
MIN_HEIGHT = 5
MIN_WIDTH = 5

# print(sys.argv[0])
if len(sys.argv) < 2:
        exit(1)  # no file handled whiten input arguments
frame = cv.imread(sys.argv[1])  #sys.argv[1]
plate = clp.cropplatefromimage(frame)

if plate is None:
    plate = frame
platelicense = ''
while DISTANCE_SIZE <= MAX_DISTANCE_SIZE:
    platelicense = dc.detectcharfromplate(plate, MIN_HW_RATIO, DISTANCE_SIZE, MIN_HEIGHT, MIN_WIDTH)
    if chkplt.checkplate(platelicense):
        break
    DISTANCE_SIZE = DISTANCE_SIZE + DISTANCE_STEP

print(str(platelicense))
sys.stdout.flush()
# cv.imshow('plate', plate)
# cv.waitKey(0)