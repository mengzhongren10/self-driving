import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math

#image = mpimg.imread('test_images/solidWhiteRight.jpg')
#
#print('This image is:',type(image),'with dimensions:',image.shape)
#plt.imshow(image)



def grayscale(img):
    #使图像变为灰度
    return cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

def canny(img,low_threshold,high_threshold):
    """Applies the Canny transform"""
    #这个是用于边缘检测的  需要处理的原图像，该图像必须为单通道的灰度图
    #其中较大的阈值2用于检测图像中明显的边缘，但一般情况下检测的效果不会那么完美，
    #边缘检测出来是断断续续的。
    #所以这时候用较小的第一个阈值用于将这些间断的边缘连接起来。
    return cv2.Canny(img,low_threshold,high_threshold)

def gaussian_blur(img,kernel_size):
    #在某些情况下，需要对一个像素的周围的像素给予更多的重视。
    #因此，可通过分配权重来重新计算这些周围点的值。
    #这可通过高斯函数（钟形函数，即喇叭形数）的权重方案来解决。
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img,(kernel_size,kernel_size),0)

def region_of_interest(img,vertices):
    
    #defining a blank mask to start with
    mask = np.zeros_like(img)
    
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,)*channel_count
    else:
        ignore_mask_color = 255
    
    #该函数填充了一个有多个多边形轮廓的区域
    cv2.fillPoly(mask,vertices,ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img,mask)
    return masked_image

def draw_lines(img,lines,color=[255,0,0],thickness = 2):
    #这个函数我也不是很理解，下面这种写法是最基本的，可以试试，其实效果并不是很好
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

 #下面是高阶的draw_lines()的写法，供参考，我是无法理解，太复杂了
#     imshape = img.shape 
    
#     slope_left=0
#     slope_right=0
#     leftx=0
#     lefty=0
#     rightx=0
#     righty=0
#     i=0
#     j=0
    
    
#     for line in lines:
#         for x1,y1,x2,y2 in line:
#             slope = (y2-y1)/(x2-x1)
#             if slope >0.1: #Left lane and not a straight line
#                 # Add all values of slope and average position of a line
#                 slope_left += slope 
#                 leftx += (x1+x2)/2
#                 lefty += (y1+y2)/2
#                 i+= 1
#             elif slope < -0.2: # Right lane and not a straight line
#                 # Add all values of slope and average position of a line
#                 slope_right += slope
#                 rightx += (x1+x2)/2
#                 righty += (y1+y2)/2
#                 j+= 1
#     # Left lane - Average across all slope and intercepts
#     if i>0: # If left lane is detected
#         avg_slope_left = slope_left/i
#         avg_leftx = leftx/i
#         avg_lefty = lefty/i
#         # Calculate bottom x and top x assuming fixed positions for corresponding y
#         xb_l = int(((int(0.97*imshape[0])-avg_lefty)/avg_slope_left) + avg_leftx)
#         xt_l = int(((int(0.61*imshape[0])-avg_lefty)/avg_slope_left)+ avg_leftx)
 
#     else: # If Left lane is not detected - best guess positions of bottom x and top x
#         xb_l = int(0.21*imshape[1])
#         xt_l = int(0.43*imshape[1])
    
#     # Draw a line
#     cv2.line(img, (xt_l, int(0.61*imshape[0])), (xb_l, int(0.97*imshape[0])), color, thickness)
    
#     #Right lane - Average across all slope and intercepts
#     if j>0: # If right lane is detected
#         avg_slope_right = slope_right/j
#         avg_rightx = rightx/j
#         avg_righty = righty/j
#         # Calculate bottom x and top x assuming fixed positions for corresponding y
#         xb_r = int(((int(0.97*imshape[0])-avg_righty)/avg_slope_right) + avg_rightx)
#         xt_r = int(((int(0.61*imshape[0])-avg_righty)/avg_slope_right)+ avg_rightx)
    
#     else: # If right lane is not detected - best guess positions of bottom x and top x
#         xb_r = int(0.89*imshape[1])
#         xt_r = int(0.53*imshape[1])
    
#     # Draw a line    
#     cv2.line(img, (xt_r, int(0.61*imshape[0])), (xb_r, int(0.97*imshape[0])), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), min_line_len, max_line_gap)
    
    
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    #lines = cv2.HoughLinesP(img,rho,theta,threshold,np.array([]),minLineLength=min_line_len,maxLineGap=max_line_gap)
    #lines = cv2.HoughLinesP(img,rho,theta,threshold,np.array([]),min_line_len,max_line_gap)
    
                           
    #霍夫变换线 用来测试直线的
    #函数cv2.HoughLinesP()是一种概率直线检测
    #我们知道，原理上讲hough变换是一个耗时耗力的算法，尤其是每一个点计算，
    #即使经过了canny转换了有的时候点的个数依然是庞大的，这个时候我们采取一种概率挑选机制，
    #不是所有的点都计算，而是随机的选取一些个点来计算，相当于降采样了
    #这样的话我们的阈值设置上也要降低一些。在参数输入输出上，输入不过多了两个参数：
    #minLineLengh(线的最短长度，比这个短的都被忽略)和MaxLineCap
    #（两条直线之间的最大间隔，小于此值，认为是一条直线）。
    
    line_img = np.zeros((img.shape[0],img.shape[1],3),dtype=np.uint8)
    draw_lines(line_img,lines)
    return line_img

def weighted_img(img,inital_img,α=0.8, β=1., γ=0.):
#   """
#    `img` is the output of the hough_lines(), An image with lines drawn on it.
#    Should be a blank image (all black) with lines drawn on it.
#    
#    `initial_img` should be the image before any processing.
#    
#    The result image is computed as follows:
#    
#    initial_img * α + img * β + γ
#    NOTE: initial_img and img must be the same shape!
#   """！
    
    
    #划线显示权重，α越大 背景图越清楚，β越大，线在图像上显示越深
    return cv2.addWeighted(inital_img,α,img,β,γ)
    
import os
#os.listdir("test_images/")

def line_detect(image):
    gray = grayscale(image)
    
    kernel_size = 5
    blur_gray = gaussian_blur(gray,kernel_size)
    
    low_threshold = 10
    high_threshold = 150
    edges = canny(blur_gray,low_threshold,high_threshold)
    
    imshape = image.shape
    vertices = np.array([[(0,imshape[0]),(int(0.45*imshape[1]),int(0.6*imshape[0])),
                          (int(0.6*imshape[1]),int(0.6*imshape[0])), (imshape[1],imshape[0])]], dtype=np.int32)
    masked_edges = region_of_interest(edges,vertices)
    
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 40 #minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments

    #threshod: 累加平面的阈值参数，int类型，超过设定阈值才被检测出线段，值越大，
    #基本上意味着检出的线段越长，检出的线段个数越少。根据情况推荐先用100试试
    #minLineLength：线段以像素为单位的最小长度，根据应用场景设置 
    #maxLineGap：同一方向上两条线段判定为一条线段的最大允许间隔（断裂），
    #超过了设定值，则把两条线段当成一条线段
    #，值越大，允许线段上的断裂越大，越有可能检出潜在的直线段
    
    line_image = hough_lines(masked_edges,rho,theta,threshold,min_line_length,max_line_gap)
    
    result = weighted_img(line_image,image,α=0.8, β=1.)
    
    return edges,masked_edges,result

import glob
new_path = os.path.join("data/test_images/","*.jpg")

for infile in glob.glob(new_path):
    image = mpimg.imread(infile)
    edges,masked_edges,result = line_detect(image)
    
    plt.figure(figsize=(20,10))
    #fig = plt.figure()
    plt.subplot(221)
    plt.title("original image")
    plt.imshow(image)
    
    plt.subplot(222)
    plt.title("canny")
    plt.imshow(edges,cmap = "gray")
    
    plt.subplot(223)
    plt.title("masked image")
    plt.imshow(masked_edges,cmap = "gray")
    
    plt.subplot(224)
    plt.title("result")
    plt.imshow(result)
