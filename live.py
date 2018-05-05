import numpy as np
import cv2
#import cPickle as pickle
from time import time

# Load trained Model
#print("Loading trained SVM Model and PCA information...")
#PCA = pickle.load(open( "PCA.dat", "rb" ))
#SVM = pickle.load(open( "SVM.dat", "rb" ))
#print("Complete")

# Capture video from computer camera
cam = cv2.VideoCapture(0)

# Define gesture names
ges = dict()
ges[0] = '0'
ges[1] = '1'
ges[2] = '2'
ges[3] = '3'
ges[4] = '4'
ges[5] = '5'
ges[6] = '6'
ges[7] = '7'
ges[8] = '8'
ges[9] = '9'
ges[10] = '10'
ges[11] = '11'
ges[12] = '12'
ges[13] = '13'
ges[14] = '14'
ges[15] = '15'
ges[16] = '16'
ges[17] = '17'
ges[18] = '18'
ges[19] = '19'
ges[20] = '20'
ges[21] = '21'
ges[22] = '22'
ges[23] = '23'
ges[24] = '24'
ges[25] = '25'
ges[26] = '26'

# Set up some storage variables
t = time()
seq_len = 18

# Run program till q is pressed
while(True):

	# Capture frame-by-frame
	ret, frame = cam.read()

	# Set up input for model


	# Get model output prediction


	# Displat output prediction overlayed on image frame
	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.imshow('preview',frame)
	#cv2.putText(frame, ges[pred],(40,40), font, 1,(255,0,0),2)
	#cv2.waitKey()

	# Print time taken per loop
	print(time() - t)
	t = time()

	# Break loop
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cam.release()
cv2.destroyAllWindows()
