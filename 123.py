import cv2
import numpy as np
from glob import glob

'''def rectify(h):
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

	return warp'''


'''img_mask = 'trainning_images/*.jpg'
img_names = glob(img_mask)
for item in img_names:
    print item
    image = cv2.imread(item)
    image_corp = getCards(image)
    cv2.imshow('test',image_corp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #item_modify = item.split('/')[1].split('.')[0]
    #image = cv2.resize(image,(200,300))
    cv2.imwrite('new/%s'%item,image_corp)'''

    
def findCard(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(1,1),1000)
    flag, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY) 

    _, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea,reverse=True)  


    peri = cv2.arcLength(contours[0],True)
    h = cv2.approxPolyDP(contours[0],0.02*peri,True)
    x,y,w,height = cv2.boundingRect(contours[0])
    #print ('h:', h)
    h = h.reshape((4,2))
    pts = np.float32(h)
    #print ('pts:', pts)
    #print ('pts:',pts)
    #h = h.reshape((4,2))
    #pts = h.reshape((4,2))
    #hnew = np.zeros((4,2),dtype = np.float32)
    temp_rect = np.zeros((4,2),dtype = np.float32)
    
    #s = np.sum(pts, axis = 2)
    s = pts.sum(1)
    #print ('s:',s)
    
    tl = pts[np.argmin(s)]
    #print ('t1:',tl)
    br = pts[np.argmax(s)]

    diff = np.diff(pts, axis = 1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]


    if w <= 0.8*height: # If card is vertically oriented
        temp_rect[0] = tl
        temp_rect[1] = tr
        temp_rect[2] = br
        temp_rect[3] = bl

    if w >= 1.2*height: # If card is horizontally oriented
        temp_rect[0] = bl
        temp_rect[1] = tl
        temp_rect[2] = tr
        temp_rect[3] = br



    if w > 0.8*height and w < 1.2*height: #If card is diamond oriented
    # If furthest left point is higher than furthest right point,
    # card is tilted to the left.
        #print ('pts101:', pts[1][1])
        #print ('pts301:', pts[3][1])
        #print ('pts10:', pts[1])
        if pts[1][1] <= pts[3][1]:
            # If card is titled to the left, approxPolyDP returns points
            # in this order: top right, top left, bottom left, bottom right
            temp_rect[0] = pts[1] # Top left
            temp_rect[1] = pts[0] # Top right
            temp_rect[2] = pts[3] # Bottom right
            temp_rect[3] = pts[2] # Bottom left

        # If furthest left point is lower than furthest right point,
        # card is tilted to the right
        if pts[1][1] > pts[3][1]:
            # If card is titled to the right, approxPolyDP returns points
            # in this order: top left, bottom left, bottom right, top right
            temp_rect[0] = pts[0] # Top left
            temp_rect[1] = pts[3] # Top right
            temp_rect[2] = pts[2] # Bottom right
            temp_rect[3] = pts[1] # Bottom left

    # box = np.int0(approx)
    # cv2.drawContours(im,[box],0,(255,255,0),6)
    # imx = cv2.resize(im,(1000,600))
    # cv2.imshow('a',imx)      

    des = np.array([[0,0],[449,0],[449,449],[0,449]],np.float32)

    transform = cv2.getPerspectiveTransform(temp_rect,des)
    warp = cv2.warpPerspective(img,transform,(450,450))


    return warp



img = cv2.imread('trainning_images/test4.jpg')
warp = getCards(img)
cv2.imshow('The card after rotation',warp)
#cv2.imshow('test',trainCards()['Spade of Eight'])
cv2.waitKey(0)
cv2.destroyAllWindows()