import numpy as np
import cv2

def enhance_image(img):
    img = unsharp_image(img)
    #img = adjust_gamma(img, 0.5)
    return img

def unsharp_image(img):
    gaussian_3 = cv2.GaussianBlur(img, (9,9), 10.0)
    unsharp_image = cv2.addWeighted(img, 2.0, gaussian_3, -1.0, 0, img)
    return unsharp_image

def adjust_gamma(img, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(img, table)
