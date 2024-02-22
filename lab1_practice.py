# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 23:02:22 2024

@author: Asus
"""

import cv2
import numpy as np

img=cv2.imread('D:\Anaconda\Bishmillah\lena.png',cv2.IMREAD_GRAYSCALE)
newimg=cv2.copyMakeBorder(img, top=1, bottom=1, left=1, right=1, borderType=cv2.BORDER_REPLICATE)
kernel=np.array(([0,-1,0],[-1,5,-1],[0,-1,0]),np.float32)

result=np.zeros((img.shape[0],img.shape[1]),dtype="float32")


v=3//2

for i in range(v,newimg.shape[0]-v):
    for j in range(v,newimg.shape[1]-v):
        sum=0
        for p in range(-v,v+1):
            for q in range(-v,v+1):
                sum=sum+kernel[p+v][q+v]*newimg[i-p][j-q]
        
           
        result[i-v][j-v]=sum

cv2.imshow('Tuli',result)
cv2.waitKey(0)
cv2.destroyAllWindows()
                