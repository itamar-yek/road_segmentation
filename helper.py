import numpy as np
import cv2
# import scipy # maybe unnecessary

def convert2bin(img, patch_size= (160, 80), end_idx= 430, kernel_morph= (3, 3), kenrel_maj= 3):
    '''
    inputs:
        img = BGR image, in range [0,255] (output of cv2.read or cv2.imread, 3d np-array)
        end_idx = the y value for the end of the main patch, default value is compatible to the image from the dual lense camera with no pre proccessing.
        patch_size = the main patch size - tuple, default- 160x80
        kernel_morph = the size of the morphological operations (dilate + erode) kernel - tuple, default- 3x3
        kernel_maj = the size of the majority kernel (implemented by median filter) - integer, default- 3x3
        
    output-
        seg = segmented image where 0 represents road, and 255 represent no-road (2d np-array)
    '''
    kernel_morph = np.ones(kernel_morph)/(kernel_morph[1]*kernel_morph[0])
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


def where2go(vid, stop_R, stop_L, angles= (30, 60, 90)):
    '''
    inputs:
        vid = the camera feed- output of cv2.videoCapture()
        stop_L = the left flag, 1 for deviation to the left and 0 otherwise
        stop_R = the left flag, 1 for deviation to the right and 0 otherwise
        
        angles = array of angles to check in degrees - np array, default [30, 60, 90] 
        
    outputs:
        inst = the "best" angle to go in degrees
    ''' 
    print("ready for scan")
    side = 'Right'
    if stop_L:
        side = 'Left'
    
    values = []
    for angle in angles:
        x = input("rotate to:" + str(angle) + 'degrees to the ' + side + 'when ready, press enter')
        ret, frame = vid.read()
        frame = np.copy(frame[:, frame.shape[1] // 2:, :])
        segment = convert2bin(frame)
        main_patch = clac_main_patch(segment)
        values.append(main_patch)
    values = np.array(values)
    chosen = angles[np.argmax(values)]
    return chosen

def clac_main_patch(seg, patch= (220, 420, 240, 360)):
    '''
    inputs:
        seg = segmented image where 0 represents road, and 255 represent no-road (2d np-array)
        patch = the x and y coordinates of the patch - tuple (y_s, y_f, x_s, x_f)
    
    output:
        main_patch = the "amount" of road pixels in the specified patch
    '''
    main_patch = np.copy(seg[patch[0]:patch[1], patch[2]:patch[3]])
    return np.sum(main_patch)

def side_patches(seg, L_patch = (360, 432, 60, 210), R_patch = (360, 456, 450, 600)):
    ''''
    inputs:
        seg = segmented image where 0 represents road, and 255 represent no-road (2d np-array)    
        L_patch = the x and y coordinates of the left patch - tuple (y_s, y_f, x_s, x_f) 
        R_patch = the x and y coordinates of the right patch - tuple (y_s, y_f, x_s, x_f) 
        
    outputs:
        Left_road_percent = the percent of road pixels among the left patch
        Right_road_percent = the percent of road pixels among the right patch
    '''
    Left = np.copy(seg[L_patch[0]:L_patch[1], L_patch[2]:L_patch[3]])
    Right = np.copy(seg[R_patch[0]:R_patch[1], R_patch[2]:R_patch[3]])
    
    Left_road_percent = (1 - np.sum(Left) / (Left.size * 255)) * 100
    Right_road_percent = (1 - np.sum(Right) / (Right.size * 255)) * 100
    
    return Left_road_percent, Right_road_percent
