import cv2
import numpy as np
from glob import glob


def trainCards():
	print '********************** Start Trainning **********************'

	trainning = {}

	img_mask = 'trainset/trainning_images/*.jpg'
	img_names = glob(img_mask)
	for item in img_names:
		image = cv2.imread(item)
		item_modify = item.split('/')[2].split('.')[0]
		image = cv2.resize(image,(450,450))
   		trainning[item_modify] = preprocessimg(image)
   	print '********************** There are ',len(trainning),' cards are trainned **********************'
   	return trainning

def preprocessimg(img):
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  	blur = cv2.GaussianBlur(gray,(5,5),0)
  	thresh = cv2.adaptiveThreshold(blur,255,1,1,11,1)
  	return thresh


def rectify(h):
  h = h.reshape((4,2))
  hnew = np.zeros((4,2),dtype = np.float32)

  add = h.sum(1)
  hnew[0] = h[np.argmin(add)]
  hnew[2] = h[np.argmax(add)]
   
  diff = np.diff(h,axis = 1)
  hnew[1] = h[np.argmin(diff)]
  hnew[3] = h[np.argmax(diff)]

  return hnew


def getCards(im):
	gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray,(1,1),1000)
	flag, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY) 
	   
	dummy, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

	contours = sorted(contours, key=cv2.contourArea,reverse=True)  


	peri = cv2.arcLength(contours[0],True)
	approx = rectify(cv2.approxPolyDP(contours[0],0.02*peri,True))

	# box = np.int0(approx)
	# cv2.drawContours(im,[box],0,(255,255,0),6)
	# imx = cv2.resize(im,(1000,600))
	# cv2.imshow('a',imx)      

	h = np.array([ [0,0],[449,0],[449,449],[0,449] ],np.float32)

	transform = cv2.getPerspectiveTransform(approx,h)
	warp = cv2.warpPerspective(im,transform,(450,450))

	return warp

def cardDiff(img1, img2):
    image1 = cv2.GaussianBlur(img1,(5,5),5)
    image2 = cv2.GaussianBlur(img2,(5,5),5)   
    diff = cv2.absdiff(image1,image2)  
    diff = cv2.GaussianBlur(diff,(5,5),5)    
    flag, diff = cv2.threshold(diff, 127, 255, cv2.THRESH_BINARY) 

    return np.sum(diff)

    '''# Initiate SIFT detector
    #sift = cv2.SIFT()
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    return len(good)'''

def matchCards(test_img, train_imgs):
    #test_features = preprocessimg(test_img)
    diff_dic = {}
    for label, train_features in train_imgs.items():
        diff_dic[label] = cardDiff(test_img, train_features)
    return sorted(diff_dic.keys(), key=diff_dic.get, reverse=True)


    '''diff_list = []
    for i, train_features in enumerate(train_imgs.values()):
        diff = cardDiff(train_features, test_features)
        diff_dict[i] = diff 
    sorted_d = sorted(d.items(), key=operator.itemgetter(0))
    testsorted_d[0].keys()'''



    

    #return sorted(training.values(), key=lambda x:imgdiff(x[1],features))[0][0] 

img = cv2.imread('trainning_images/test8.jpg')
test_img = preprocessimg(getCards(img))
print matchCards(test_img,trainCards())

cv2.imshow('The card after rotation',test_img)
cv2.imshow('trainset',trainCards()['Diamond of Seven'])
cv2.waitKey(0)
cv2.destroyAllWindows()

'''cv2.imshow('test',trainCards()['Spade of Six'])
cv2.waitKey(0)
cv2.destroyAllWindows()'''
