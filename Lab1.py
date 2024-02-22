# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 23:44:21 2024

@author: Asus
"""

import cv2
import numpy as np
image=cv2.imread('D:\Image Lab\Bishmillah\lena.png',cv2.IMREAD_GRAYSCALE)
cv2.imshow('input',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
border_image=cv2.copyMakeBorder(image, top=1, bottom=1, left=1, right=1, borderType=cv2.BORDER_CONSTANT)
cv2.imshow('Bordered_Image',border_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

kernel =np.array([[-1,-2,-1],[0,0,0],[1,2,1]
    ])
padding=(kernel.shape[0]-1)//2
output_image=np.zeros((image.shape[0],image.shape[1]),dtype=np.uint8)
for i in range(padding,border_image.shape[0]-padding):
    for j in range(padding,border_image.shape[1]-padding):
        sum=0
        for p in range(-padding,padding+1):
            for q in range(-padding,padding+1):
                sum+=kernel[p+padding][q+padding]*border_image[i-p][j-q]
                
        output_image[i-padding][j-padding]=sum

cv2.normalize(output_image, output_image,0,255,cv2.NORM_MINMAX)
output_image=np.round(output_image).astype(np.uint8)
cv2.imshow('output',output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
        






