import numpy as np
import cv2
from glob import glob

import time
from matplotlib import pyplot as plt


# Adaptive threshold levels
BKG_THRESH = 60
CARD_THRESH = 30

CARD_MAX_AREA = 12000
CARD_MIN_AREA = 2500


def trainCards():
	
	trainning = {}

	img_mask = 'trainning_images/*.jpg'
	img_names = glob(img_mask)
	for item in img_names:
		image = cv2.imread(item)
		item_modify = item.split('/')[1].split('.')[0]
   		trainning[item_modify] = (preprocessimg(image))
   	print trainning
   	return trainning

	'''for (i,image_file) in enumerate(glob.iglob('/trainning_images/')):
			trainning[i] = (image_file,preprocessimg(image_file))'''

def preprocessimg(img):
	#image = cv2.resize(image,(1280,720))
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray,(5,5),0)

	img_w, img_h = np.shape(image)[:2]
	bkg_level = gray[int(img_h/100)][int(img_w/2)]
	thresh_level = bkg_level + BKG_THRESH
	retval, thresh_img = cv2.threshold(blur,thresh_level,255,cv2.THRESH_BINARY)

	return thresh_img
	
def matchCards(img1,img2):
	pass

cv2.imshow('test',trainCards()['Spade of Six'])
cv2.waitKey(0)
cv2.destroyAllWindows()
#print trainCards()[6]
