import numpy as np
import cv2
import scipy # maybe unnecessary

def convert2bin(img, patch_size= (160, 80), end_idx= 430, kernel_morph= (3, 3), kenrel_maj= (3, 3)):
    '''
    inputs:
        img = BGR image, in range [0,255] (output of cv2.read or cv2.imread, 3d np-array)
        end_idx = the y value for the end of the main patch, default value is compatible to the image from the dual lense camera with no pre proccessing.
        patch_size = the main patch size - tuple, default- 160x80
        kernel_morph = the size of the morphological operations (dilate + erode) kernel - tuple, default- 3x3
        kernel_maj = the size of the majority kernel (implemented by median filter) - tuple, default- 3x3
        
    output-
        seg = segmented image where 0 represents road, and 255 represent no-road (2d np-array)
    '''
    kernel_morph = np.ones(kernel_morph)/(kernel_morph[1]*kernel_morph[0])
    # kenrel_maj = np.ones(kenrel_maj)
    sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1]  # take the saturation channel from the hsv matrix
    patch = sat[(end_idx-patch_size[1]):end_idx, (sat.shape[1]//2 - patch_size[0]//2):sat.shape[1]//2 + patch_size[0]//2] 
    # the above line takes the patch to be symetric around y axis and to end on end_idx
    sat_mean = np.mean(patch) #calculate the mean of the patch
    thresh = 1.05 * sat_mean + 35.22  # calculate the threshold
    ret, bin_img = cv2.threshold(sat, thresh, 255, cv2.THRESH_BINARY)  # apply threshold 
    # improve the segmentation using morphological operators

    seg = cv2.medianBlur(bin_img, kenrel_maj)
    seg = cv2.medianBlur(seg, kenrel_maj)
    seg = cv2.dilate(seg, kernel_morph, iterations=2)
    seg = cv2.erode(seg, kernel_morph, iterations=1)
    return seg


def where2go():
    '''
    check 4 different angles and decide where to go
    '''
    return

def angle_patch():
    '''
    returns the angle patch value
    '''
    return

def side_patches():
    ''''
    returnnsn the values of the patches, left and right
    '''
    return


