import numpy as np
import cv2
from glob import glob

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
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  	blur = cv2.GaussianBlur(gray,(5,5),2 )
  	thresh = cv2.adaptiveThreshold(blur,255,1,1,11,1)
  	return thresh


def matchCards(img1,img2):
	pass

cv2.imshow('test',trainCards()['Spade of Six'])
cv2.waitKey(0)
cv2.destroyAllWindows()
#print trainCards()[6]
