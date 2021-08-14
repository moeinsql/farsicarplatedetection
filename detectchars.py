import cv2
import math
import numpy as np
import loadtrainedknn
from imutils import contours as sc

# parameters
GAUSSIAN_SMOOTH_FILTER_SIZE = (3, 3) #(5, 5)
ADAPTIVE_THRESH_BLOCK_SIZE = 15 #19
ADAPTIVE_THRESH_WEIGHT = 9  #9

RESIZED_IMAGE_WIDTH = 150  #150
RESIZED_IMAGE_HEIGHT = 30  #30

RESIZED_CHAR_IMAGE_WIDTH = 20
RESIZED_CHAR_IMAGE_HEIGHT = 30


def distanceBetweenChars(firstChar, secondChar):
	x1, y1, w1, h1 = cv2.boundingRect(firstChar)
	x2, y2, w2, h2 = cv2.boundingRect(secondChar)
	X = abs( ((x1 + x1 + w1) / 2) - ((x2 + x2 + w2) / 2) )
	Y = abs( ((y1 + y1 + h1) / 2) - ((y2 + y2 + h2) / 2) )
	return math.sqrt((X ** 2) + (Y ** 2))
# end function

###################################################################################################
# use basic trigonometry (SOH CAH TOA) to calculate angle between chars
def angleBetweenChars(firstChar, secondChar):
	x1, y1, w1, h1 = cv2.boundingRect(firstChar)
	x2, y2, w2, h2 = cv2.boundingRect(secondChar)
	fltAdj = float(abs(((x1 + x1 + w1) / 2) - ((x2 + x2 + w2) / 2)))
	fltOpp = float(abs(((y1 + y1 + h1) / 2) - ((y2 + y2 + h2) / 2)))

	if fltAdj != 0.0:                           # check to make sure we do not divide by zero if the center X positions are equal, float division by zero will cause a crash in Python
		fltAngleInRad = math.atan(fltOpp / fltAdj)      # if adjacent is not zero, calculate angle
	else:
		fltAngleInRad = 1.5708                          # if adjacent is zero, use this as the angle, this is to be consistent with the C++ version of this program
	# end if

	fltAngleInDeg = fltAngleInRad * (180.0 / math.pi)       # calculate angle in degrees

	return fltAngleInDeg
# end function
def isPossibleChar(cnt, MIN_HW_RATIO, MIN_HEIGHT, MIN_WIDTH):
	x, y, w, h = cv2.boundingRect(cnt)
	if w < MIN_WIDTH and h < MIN_HEIGHT:
		return False

	hwratio = min(h,w) / max(h, w)
	if hwratio <= MIN_HW_RATIO:
		return False
	return True

def detectcharfromplate(plate, MIN_HW_RATIO, MAX_DISTANCE_SIZE, MIN_HEIGHT, MIN_WIDTH):
	# main detect char
	# load knn trained model
	isloaded, knnmodel = loadtrainedknn.loadKNNDataAndTrainKNN()
	if not isloaded:
		print("knn load err")
		exit(1)

	image_resized = cv2.resize(plate, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
	imgGrayscale = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
	imgBlurred = cv2.GaussianBlur(imgGrayscale, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)
	imgThresh = cv2.adaptiveThreshold(imgBlurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
									  ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)
	contours, hierarchy = cv2.findContours(imgThresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	# for cnt in contours:
	# 	x,y,w,h = cv2.boundingRect(cnt)
	#  	#bound the images
	# 	cv2.rectangle(image, (x,y),(x+w,y+h),(0,255,0),3)

	i = 0
	strChars = ""
	contours, _ = sc.sort_contours(contours, method="left-to-right")

	for cnt in contours:
		x, y, w, h = cv2.boundingRect(cnt)
		if isPossibleChar(cnt, MIN_HW_RATIO, MIN_HEIGHT, MIN_WIDTH):
			# cv2.imwrite("res/" + str(i) + ".jpg", imgThresh[y:y + h, x:x + w])
			imgROI = imgThresh[y:y + h, x:x + w]
			imgROIResized = cv2.resize(imgROI, (RESIZED_CHAR_IMAGE_WIDTH,
												RESIZED_CHAR_IMAGE_HEIGHT))  # resize image, this is necessary for char recognition
			npaROIResized = imgROIResized.reshape(
				(1, RESIZED_CHAR_IMAGE_WIDTH * RESIZED_CHAR_IMAGE_HEIGHT))  # flatten image into 1d numpy array
			npaROIResized = np.float32(npaROIResized)  # convert from 1d numpy array of ints to 1d numpy array of floats
			retval, npaResults, neigh_resp, dists = knnmodel.findNearest(npaROIResized,
																		 k=1)  # finally we can call findNearest !!!
			strCurrentChar = str(chr(int(npaResults[0][0])))  # get character from results
			# print(strCurrentChar , float(dists[0][0]))
			if float(dists[0][0]) < MAX_DISTANCE_SIZE:
				strChars = strChars + strCurrentChar
			i = i + 1

	return strChars
	# cv2.namedWindow('BindingBox', cv2.WINDOW_NORMAL)
	# cv2.imshow('BindingBox', imgThresh)
	# cv2.waitKey()