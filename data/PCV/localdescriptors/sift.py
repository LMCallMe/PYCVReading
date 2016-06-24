#encoding:utf-8

from PIL import Image
import os
import cv2
from numpy import *
from pylab import *

def read_features_from_file(imgname):
    im = cv2.imread(imgname)
    im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT()

    kp, desc = sift.detectAndCompute(im,None)
    return kp, desc

def plot_features(im,kp,circle=False):
    """ show image with features. input: im (image as array), 
        locs (row, col, scale, orientation of each feature) """
    if circle:
        img=cv2.drawKeypoints(im,kp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    else:
        img=cv2.drawKeypoints(im,kp)
        
    imshow(img)
    axis('off')


def match(desc1,desc2):
    """ 对第一副图像的每个描述子,找到在第二幅图像中的匹配 OpenCV 版"""
    
    dist_ratio = 0.6
    desc1_size = desc1.shape
    
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1,desc2, k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < dist_ratio*n.distance:
            good.append(m)
    
    # 仿照原书的输出
    matchscores = zeros((desc1_size[0],1))
    for m in good:
        matchscores[m.queryIdx] = m.trainIdx

    return matchscores


def appendimages(im1,im2):
    """ return a new image that appends the two images side-by-side."""
    
    #select the image with the fewest rows and fill in enough empty rows
    rows1 = im1.shape[0]    
    rows2 = im2.shape[0]
    
    if rows1 < rows2:
        im1 = concatenate((im1,zeros((rows2-rows1,im1.shape[1]))), axis=0)
    elif rows1 > rows2:
        im2 = concatenate((im2,zeros((rows1-rows2,im2.shape[1]))), axis=0)
    #if none of these cases they are equal, no filling needed.
    
    return concatenate((im1,im2), axis=1)


def plot_matches(im1,im2,kp1,kp2,matchscores,show_below=True):
    """ OpwnCV 版:
        show a figure with lines joining the accepted matches
        input: im1,im2 (images as arrays), locs1,locs2 (location of features), 
        matchscores (as output from 'match'), show_below (if images should be shown below). """
    
    im3 = appendimages(im1,im2)
    if show_below:
        im3 = vstack((im3,im3))
    
    # show image
    imshow(im3)
    
    # draw lines for matches
    cols1 = im1.shape[1]
    for i in range(len(matchscores)):
        if matchscores[i] > 0:
            pt1 = kp1[i].pt
            pt2 = kp2[int(matchscores[i,0])].pt
            plot([pt1[0],pt2[0]+cols1], [pt1[1], pt2[1]], 'c')
    axis('off')


def match_twosided(desc1,desc2):
    """ two-sided symmetric version of match(). """
    
    matches_12 = match(desc1,desc2)
    matches_21 = match(desc2,desc1)
    
    ndx_12 = matches_12.nonzero()[0]
    
    #remove matches that are not symmetric
    for n in ndx_12:
        if matches_21[int(matches_12[n])] != n:
            matches_12[n] = 0
    
    return matches_12


if __name__ == "__main__":
    
    process_image('box.pgm','tmp.sift')
    l,d = read_features_from_file('tmp.sift')
    
    im = array(Image.open('box.pgm'))
    figure()
    plot_features(im,l,True)
    gray()
    
    process_image('scene.pgm','tmp2.sift')
    l2,d2 = read_features_from_file('tmp2.sift')
    im2 = array(Image.open('scene.pgm'))    
    
    m = match_twosided(d,d2)
    figure()
    plot_matches(im,im2,l,l2,m)

    gray()
    show()
    
    
    
