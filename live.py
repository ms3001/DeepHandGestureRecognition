from collections import OrderedDict
import cv2
import numpy as np
import pickle
from model import ConvColumn
from PIL import Image
from time import time
import torch
from torch.autograd import Variable
from torchvision.transforms import *


# Load trained Model
#print("Loading trained SVM Model and PCA information...")
#PCA = pickle.load(open( "PCA.dat", "rb" ))
#SVM = pickle.load(open( "SVM.dat", "rb" ))
#print("Complete")

# Capture video from computer camera
cam = cv2.VideoCapture(0)

# Define gesture names
ges = dict()
ges[0] = 'Swiping left'
ges[1] = 'Swiping right'
ges[2] = 'Swiping down'
ges[3] = 'Swiping up'
ges[4] = 'Pushing hand away'
ges[5] = 'Pulling hand down'
ges[6] = 'Sliding Two Fingers Left'
ges[7] = 'Sliding Two Fingers Right'
ges[8] = 'Sliding Two Fingers Down'
ges[9] = 'Sliding Two Fingers Up'
ges[10] = 'Pushing Two Fingers Away'
ges[11] = 'Pulling Two Fingers In'
ges[12] = 'Rolling Hand Forward'
ges[13] = 'Rolling Hand Backward'
ges[14] = 'Turning Hand Clockwise'
ges[15] = 'Turning Hand Counterclockwise'
ges[16] = 'Zooming In With Full Hand'
ges[17] = 'Zooming Out With Full Hand'
ges[18] = 'Zooming In With Two Fingers'
ges[19] = 'Zooming Out With Two Fingers'
ges[20] = 'Thumb Up'
ges[21] = 'Thumb Down'
ges[22] = 'Shaking Hand'
ges[23] = 'Stop Sign'
ges[24] = 'Drumming Fingers'
ges[25] = 'No gesture'
ges[26] = 'Doing other things'
ges[27] = 'waiting for more frames'

# Set up some storage variables
t = time()
seq_len = 18
imgs = []
pred = 27

# Load model
print('Loading model...')
state_dict = torch.load('model_best.pth.tar', map_location='cpu')['state_dict']

state_dict_rename = OrderedDict()
for k, v in state_dict.items():
	name = k[7:] # remove 'module.'
	state_dict_rename[name] = v

model = ConvColumn(27,(3, 3, 3))
model.load_state_dict(state_dict_rename)

transform = Compose([CenterCrop(84), ToTensor()])
print('Starting prediction')
# Run program till q is pressed
while(True):

	# Capture frame-by-frame
	ret, frame = cam.read()

	#print(np.shape(frame))

	# Set up input for model
	resized_frame = cv2.resize(frame, (149, 84))

	#print(np.shape(resized_frame))

	pre_img = Image.fromarray(resized_frame.astype('uint8'), 'RGB')

	#print(pre_img.size)
	#print(pre_img.mode)

	img = transform(pre_img)
	#print(img.size())
	imgs.append(torch.unsqueeze(img, 0))

	if len(imgs) > 18:
		imgs.pop(0)

	# Get model output prediction
	if len(imgs) == 18:
		# format data to torch
		data = torch.cat(imgs)
		data = data.permute(1, 0, 2, 3)
	#	print(data.size())
		output = model(Variable(data).unsqueeze(0))
		out = (output.data).cpu().numpy()[0]
		print('Model output:', out)
		indices = np.argmax(out)
		print('Max index:', indices)
		pred = indices


	# Displat output prediction overlayed on image frame
	
	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(frame, ges[pred],(40,40), font, 1,(0,0,0),2)
	cv2.imshow('preview',frame)

	# Print time taken per loop
	print(time() - t)
	t = time()

	# Break loop
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cam.release()
cv2.destroyAllWindows()
