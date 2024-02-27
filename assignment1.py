# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 01:48:35 2024

@author: Asus
"""

import cv2
import numpy as np
import math

def main():
    print("Assignment1_1907019")
    print("1.GrayScale Image:")
    print("2.RGB Image:")
    print("3.HSV Image:")
    print("0:Exit")
    choice = input("Enter Your Choice: ")
    choice = int(choice)
    if choice == 1:
        image=cv2.imread('Lena.jpg',cv2.IMREAD_GRAYSCALE)
        image2=cv2.imread('noisy_image.jpg',cv2.IMREAD_GRAYSCALE)
        #cv2.imshow('GrayScale',image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
    elif choice==2:
        image=cv2.imread('Lena.jpg',cv2.IMREAD_COLOR)
        image2=cv2.imread('noisy_image.jpg',cv2.IMREAD_COLOR)
        #cv2.imshow('RGB',image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
    elif choice==3:
        image=cv2.imread('Lena.jpg',cv2.COLOR_RGB2HSV)
        image2=cv2.imread('noisy_image.jpg',cv2.COLOR_RGB2HSV)
        #cv2.imshow('HSV',image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
    
    #image=cv2.resize(image,(500,500))
    print("******Smoothing Filter******")
    print("1.Gaussian Blur Filter")
    print("2.Mean Filter")
    print("*****Sharpening Filter******")
    print("3.Laplacian Filter")
    print("4.Laplacian of a Gaussian(LOG)")
    print("5.Sobel Filter")
    #print("Enter Your Choice: ")
    filter_str=input("Enter Your Choice: ")
    filter=int(filter_str)
    if filter == 1:
        sigmax=float(input("Enter the value of sigmax: "))
        sigmay=float(input("Enter the value of sigmay: "))
        kernel=get_gaussian_kernel(sigmax, sigmay)
        center_x=int(input("Kernel center_x: "))
        center_y=int(input("Kernel center_y: "))
        if choice == 1:
            result=convolution("grayscale",image,kernel,center_x,center_y)
            cv2.imshow("GrayScale_Image",image)
            cv2.imshow("Gaussian Filtered Image",result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif choice == 2:
            b1,g1,r1=cv2.split(image2)
            b1=convolution("blue",b1,kernel,center_x,center_y)
            g1=convolution("green",g1,kernel,center_x,center_y)
            r1=convolution("red",r1,kernel,center_x,center_y)
            merged=cv2.merge((b1,g1,r1))
            cv2.imshow("RGB Image",image2)
            cv2.imshow("blue",b1)
            cv2.imshow("green",g1)
            cv2.imshow("red",r1)
            
            cv2.imshow("merged",merged)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
    elif filter == 2:
        print("Enter Your Kernel Hight and Width: ")
        x=input("height: ")
        x=int(x)
        y=input("Width:  ")
        y=int(y)
        kernel=get_mean_kernel(x,y)
        center_x=int(input("Kernel center_x: "))
        center_y=int(input("Kernel center_y: "))
        if choice == 1:
            result=convolution("grayscale",image,kernel,center_x,center_y)
            cv2.imshow("GrayScale_Image",image)
            cv2.imshow("Mean Filtered Image",result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif choice == 2:
            b1,g1,r1=cv2.split(image)
            b1=convolution("blue",b1,kernel,center_x,center_y)
            g1=convolution("green",g1,kernel,center_x,center_y)
            r1=convolution("red",r1,kernel,center_x,center_y)
            merged=cv2.merge((b1,g1,r1))
            cv2.imshow("RGB Image",image)
            cv2.imshow("blue",b1)
            cv2.imshow("green",g1)
            cv2.imshow("red",r1)
            
            cv2.imshow("merged",merged)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    elif filter ==3:
        print("Enter Your Kernel Hight and Width: ")
        x=input("height: ")
        x=int(x)

        if(x==3) :
            kernel=np.array([[0, 1, 0],
                                [1, -4, 1],
                                [0, 1, 0]], dtype=np.float32)
        elif(x==5):
            kernel=np.array([ [0, 0,  1,  0, 0],
                                [0, 1,  2,  1, 0],
                                [1, 2, -16, 2, 1],
                                [0, 1,  2,  1, 0],
                                [0, 0,  1,  0, 0]], dtype=np.float32)
            
        
        center_x=int(input("Kernel center_x: "))
        center_y=int(input("Kernel center_y: "))
        if choice == 1:
            result=convolution("grayscale",image,kernel,center_x,center_y)
            cv2.imshow("GrayScale_Image",image)
            cv2.imshow("laplacian Filtered Image",result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            b1,g1,r1=cv2.split(image)
            b1=convolution("blue",b1,kernel,center_x,center_y)
            g1=convolution("green",g1,kernel,center_x,center_y)
            r1=convolution("red",r1,kernel,center_x,center_y)
            merged=cv2.merge((b1,g1,r1))
            cv2.imshow("RGB Image",image)
            cv2.imshow("blue",b1)
            cv2.imshow("green",g1)
            cv2.imshow("red",r1)
            cv2.imshow("merged",merged)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    elif filter == 4:
        print("Enter Your Kernel Hight and Width: ")
        x=input("height: ")
        x=int(x)

        if(x==3) :
            kernel=np.array([[0, 1, 0],
                                [1, -4, 1],
                                [0, 1, 0]], dtype=np.float32)
        elif(x==5):
            kernel=np.array([ [0, 0,  1,  0, 0],
                                [0, 1,  2,  1, 0],
                                [1, 2, -16, 2, 1],
                                [0, 1,  2,  1, 0],
                                [0, 0,  1,  0, 0]], dtype=np.float32)
            
        
        center_x=int(input("Kernel center_x: "))
        center_y=int(input("Kernel center_y: "))
        if choice == 1:
            result=convolution("grayscale",image,kernel,center_x,center_y)
            cv2.imshow("GrayScale_Image",image)
            cv2.imshow("laplacian Filtered Image",result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            b1,g1,r1=cv2.split(image)
            b1=convolution("blue",b1,kernel,center_x,center_y)
            g1=convolution("green",g1,kernel,center_x,center_y)
            r1=convolution("red",r1,kernel,center_x,center_y)
            merged=cv2.merge((b1,g1,r1))
            cv2.imshow("merged",merged)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    elif filter == 5:
        print("1.Horizontal Kernel")
        print("2.Vertical Kernel")
        h_v=int(input("Enter Your Choice: "))
        if h_v == 1:
            kernel=np.array([[-1, 0, 1],
                                 [-2, 0, 2], 
                                 [-1, 0, 1]])
            
        elif h_v == 2:
            kernel=np.array([[-1,-2, -1], 
                                     [0, 0, 0], 
                                     [1, 2, 1]])
        center_x=1
        center_y=1
        if choice == 1:
            result=convolution("grayscale",image,kernel,center_x=1,center_y=1)
            cv2.imshow("GrayScale_Image",image)
            cv2.imshow("Sobel_Result",result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif choice == 2:
            b1,g1,r1=cv2.split(image)
            b1=convolution("blue",b1,kernel,center_x,center_y)
            g1=convolution("green",g1,kernel,center_x,center_y)
            r1=convolution("red",r1,kernel,center_x,center_y)

            merged=cv2.merge((b1,g1,r1))
            cv2.imshow("blue",b1)
            cv2.imshow("green",g1)
            cv2.imshow("red",r1)
            cv2.imshow("merged",merged)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
#Defining Kernel    
       
def get_mean_kernel(height,width):
    mean_kernel=np.ones((height,width),dtype=np.float32)
    mean_kernel=mean_kernel/(height*width)
    return mean_kernel

def get_gaussian_kernel(sigmax,sigmay):
    height=int(sigmax*5)
    width=int(sigmay*5)
    if height%2==0:
        height+=1
    if width%2==0:
        width+=1
    gaussian_kernel=np.zeros((height,width),dtype=np.float32)
    constant=1/(2*math.pi*(sigmax*sigmay))
    height=height//2
    width=width//2
    
    for x in range(-height,height+1):
        for y in range(-width,width+1):
            exponent=-0.5*((x*x)/(sigmax*sigmax)+(y*y)/(sigmay*sigmay))
            value=constant*math.exp(exponent)
    
            gaussian_kernel[x+height][y+width]=value
    
    return gaussian_kernel
    
    
    




#Convolution Code
def convolution(s,image,kernel,center_x,center_y):
    k = kernel.shape[0] // 2
    l = kernel.shape[1] // 2
    padding_bottom = kernel.shape[0] - 1 - center_x
    padding_right = kernel.shape[1] - 1 - center_y
    img_bordered = cv2.copyMakeBorder(src=image, top=center_x, bottom=padding_bottom, left=center_y, right=padding_right,borderType=cv2.BORDER_CONSTANT)
    out = np.zeros((img_bordered.shape[0],img_bordered.shape[1]),dtype=np.uint8)

    for i in range(center_x, img_bordered.shape[0] - padding_bottom - k):
        for j in range(center_y, img_bordered.shape[1] - padding_right - l):
            res = 0
            for x in range(-k, k + 1):
                for y in range(-l, l + 1):
                    res += kernel[x + k, y + l] * img_bordered[i - x, j - y]
            out[i, j] = res 
            
    # crop image to original image
    #out = out[center_x: -padding_bottom, center_y:-padding_right]
    cv2.normalize(out, out, 0, 255, cv2.NORM_MINMAX)
    out = np.round(out).astype(np.uint8)
    #print(f"normalized {out}") 
    #out = out[p: -padding_bottom, q:-padding_right]
           
    return out
    
 
    
    
if __name__ == "__main__":
    main()
    
    
    