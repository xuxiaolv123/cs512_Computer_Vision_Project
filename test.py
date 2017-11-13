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

def preprocessimg(image):
	#image = cv2.resize(image,(1280,720))
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray,(5,5),0)

	img_w, img_h = np.shape(image)[:2]
	bkg_level = gray[int(img_h/100)][int(img_w/2)]
	thresh_level = bkg_level + BKG_THRESH
	retval, thresh_img = cv2.threshold(blur,thresh_level,255,cv2.THRESH_BINARY)

	return thresh_img


def find_card(thresh_img, image):
    # Find contours and sort their indices by contour size
    dummy, cnts, hier = cv2.findContours(thresh_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    index_sort = sorted(range(len(cnts)), key=lambda i : cv2.contourArea(cnts[i]),reverse=True)

    #print ('lens of find contours: %d'%len(cnts))

    #initialize empty sorted contour and hierarchy lists
    cnts_sort = []
    hier_sort = []
   
    for i in index_sort:
        cnts_sort.append(cnts[i])
        hier_sort.append(hier[0][i])

    card_cnt = cnts_sort[0]

    # Find perimeter of card and use it to approximate corner points
    peri = cv2.arcLength(card_cnt,True)
    approx = cv2.approxPolyDP(card_cnt,0.01*peri,True)
    pts = np.float32(approx)

    # Find width and height of card's bounding rectangle
    x,y,w,h = cv2.boundingRect(cnt)  #(x,y) be the top-left coordinate of the rectangle and (w,h) be its width and height.
    temp_rect = np.zeros((4,2), dtype = "float32")

    s = np.sum(pts, axis = 2)
  
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]

    diff = np.diff(pts, axis = -1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    if w <= 0.8*h: # If card is vertically oriented
        temp_rect[0] = tl
        temp_rect[1] = tr
        temp_rect[2] = br
        temp_rect[3] = bl

    if w >= 1.2*h: # If card is horizontally oriented
        temp_rect[0] = bl
        temp_rect[1] = tl
        temp_rect[2] = tr
        temp_rect[3] = br



    if w > 0.8*h and w < 1.2*h: #If card is diamond oriented
        # If furthest left point is higher than furthest right point,
        # card is tilted to the left.
        if pts[1][0][1] <= pts[3][0][1]:
            # If card is titled to the left, approxPolyDP returns points
            # in this order: top right, top left, bottom left, bottom right
            temp_rect[0] = pts[1][0] # Top left
            temp_rect[1] = pts[0][0] # Top right
            temp_rect[2] = pts[3][0] # Bottom right
            temp_rect[3] = pts[2][0] # Bottom left

        # If furthest left point is lower than furthest right point,
        # card is tilted to the right
        if pts[1][0][1] > pts[3][0][1]:
            # If card is titled to the right, approxPolyDP returns points
            # in this order: top left, bottom left, bottom right, top right
            temp_rect[0] = pts[0][0] # Top left
            temp_rect[1] = pts[3][0] # Top right
            temp_rect[2] = pts[2][0] # Bottom right
            temp_rect[3] = pts[1][0] # Bottom left


    maxWidth = 200
    maxHeight = 300

    # Create destination array, calculate perspective transform matrix,
    # and warp card image
    dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0, maxHeight-1]], np.float32)
    M = cv2.getPerspectiveTransform(temp_rect,dst)
    warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    #warp = cv2.cvtColor(warp,cv2.COLOR_BGR2GRAY)

    return warp


def matchCards(img1,img2):
	pass


cv2.imshow('test',trainCards()['Spade of Six'])
cv2.waitKey(0)
cv2.destroyAllWindows()
#print trainCards()[6]
