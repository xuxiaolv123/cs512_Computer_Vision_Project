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
		image = cv2.resize(image,(200,300))
   		trainning[item_modify] = (preprocessimg(image))
   	print len(trainning)
   	return trainning

	'''for (i,image_file) in enumerate(glob.iglob('/trainning_images/')):
			trainning[i] = (image_file,preprocessimg(image_file))'''

def preprocessimg(img):
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  	blur = cv2.GaussianBlur(gray,(5,5),0)
  	thresh = cv2.adaptiveThreshold(blur,255,1,1,11,1)
  	return thresh


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
    x,y,w,h = cv2.boundingRect(card_cnt)  #(x,y) be the top-left coordinate of the rectangle and (w,h) be its width and height.
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

def cardDiff(img1, img2):
    '''image1 = cv2.GaussianBlur(img1,(5,5),5)
    image2 = cv2.GaussianBlur(img2,(5,5),5)   
    diff = cv2.absdiff(image1,image2)  
    diff = cv2.GaussianBlur(diff,(5,5),5)    
    flag, diff = cv2.threshold(diff, 127, 255, cv2.THRESH_BINARY) 

    return np.sum(diff)'''

    # Initiate SIFT detector
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
    return len(good)

def matchCards(test_img, train_imgs):
    test_features = preprocessimg(test_img)
    diff_dic = {}
    for label, train_features in train_imgs.items():
        diff_dic[label] = cardDiff(test_features, train_features)
    return sorted(diff_dic.keys(), key=diff_dic.get, reverse=True)


    '''diff_list = []
    for i, train_features in enumerate(train_imgs.values()):
        diff = cardDiff(train_features, test_features)
        diff_dict[i] = diff 
    sorted_d = sorted(d.items(), key=operator.itemgetter(0))
    testsorted_d[0].keys()'''



    

    #return sorted(training.values(), key=lambda x:imgdiff(x[1],features))[0][0] 

img = cv2.imread('trainning_images/test6.jpg')
test_img = find_card(preprocessimg(img),img)
print matchCards(test_img,trainCards())

'''cv2.imshow('test',trainCards()['Spade of Six'])
cv2.waitKey(0)
cv2.destroyAllWindows()'''
#print trainCards()[6]
