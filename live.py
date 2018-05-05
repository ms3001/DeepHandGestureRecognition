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
	cv2.imshow('preview',frame)
	cv2.putText(frame, ges[pred],(40,40), font, 1,(0,0,0),2)

	# Print time taken per loop
	print(time() - t)
	t = time()

	# Break loop
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cam.release()
cv2.destroyAllWindows()
