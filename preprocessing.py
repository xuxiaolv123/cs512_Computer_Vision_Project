def preprocessing(image):
	#image = cv2.resize(image,(1280,720))
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray,(5,5),0)

	img_w, img_h = np.shape(image)[:2]
	bkg_level = gray[int(img_h/100)][int(img_w/2)]
	thresh_level = bkg_level + BKG_THRESH
	retval, thresh_img = cv2.threshold(blur,thresh_level,255,cv2.THRESH_BINARY)

	return thresh_img