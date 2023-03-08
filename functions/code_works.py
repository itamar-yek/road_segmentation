import numpy as np
import cv2
import os


def where_to_go():
    kernel = np.array([
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1]
    ], dtype='uint8') / 16
    win_x = 80
    win_y = 40
    R_up_lim = 0.75
    R_down_lim = 0.95
    L_up_lim = 0.75
    L_down_lim = 0.9

    angles = np.zeros(3)
    win2_y = 150
    print(" ready for scan")
    for i in range(angles.shape[0]):
        x = input("Press enter for "+str(i)+" angle")
        ret, frame = vid.read()
        image = np.copy(frame[:, int(frame.shape[1] / 2):, :])  # take only the left camera
        sat = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 0]  # convert to hsv
        sat_mean = np.mean(sat[int(end_idx - 2 * win_y):end_idx, int(sat.shape[1] / 2 - win_x):int(
            sat.shape[1] / 2 + win_x)])  # calculate the mean of the chosen window for the adaptive threshold
        thresh = 1.05 * sat_mean + 35.22  # calculate the threshold
        ret_2, bin_img = cv2.threshold(sat, thresh, 255, cv2.THRESH_BINARY)  # apply threshold
        dilated_img = cv2.dilate(bin_img, kernel, iterations=2)
        dilated_img = cv2.erode(dilated_img, kernel, iterations=1)
        cv2.imwrite('angle_'+str(i)+'.png', dilated_img)
        angles[i] = np.sum(dilated_img[220:420, 240:360])
    print("go to angle- "+str(np.argmin(angles)))
    x = input("Press enter to continue")




i=0
cv2.destroyAllWindows()
win_x = 80
win_y = 40
R_up_lim = 0.75
R_down_lim = 0.95
L_up_lim = 0.75
L_down_lim = 0.9
precent_thresh = 40
end_idx = 430
stop_R = False
stop_L = False
stop_counter_R = 0
stop_counter_L = 0
counter_thresh = 30

im = cv2.imread('image_1.png')
stop_image_R = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im = cv2.imread('image_2.png')
stop_image_L = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

#vid = cv2.VideoCapture(0)
vid = cv2.VideoCapture('jonathan_massege2.mp4')

# saving video
ret, frame = vid.read()
fps = 20
vid_format = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('out.mp4', vid_format, fps, (frame.shape[1], frame.shape[0]))
out_bin = cv2.VideoWriter('out_bin.mp4', vid_format, fps,
                          (frame.shape[1]//2, int(frame.shape[0])))  # maybe the zero at the end should be deleted

kernel = np.array([
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1]
], dtype='uint8') / 16

ret = True
c = 0

while ret:

    ret, frame = vid.read()  # get the image from the camera
    image = np.copy(frame[:, int(frame.shape[1] / 2):, :])  # take only the left camera
    h_s_v = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # convert to hsv
    sat = h_s_v[:, :, 1]  # take the saturation channel only
    sat_mean = np.mean(sat[int(end_idx - 2 * win_y):end_idx, int(sat.shape[1] / 2 - win_x):int(
        sat.shape[1] / 2 + win_x)])  # calculate the mean of the chosen window for the adaptive threshold
    thresh = 1.05 * sat_mean + 35.22  # calculate the threshold
    ret_2, bin_img = cv2.threshold(sat, thresh, 255, cv2.THRESH_BINARY)  # apply threshold
    temp = np.zeros((bin_img.shape[0], bin_img.shape[1], 3), dtype='uint8')
    temp2 = np.zeros((bin_img.shape[0], bin_img.shape[1], 3), dtype='uint8')
    # improve the segmentation using morphological operators
    dilated_img = cv2.dilate(bin_img, kernel, iterations=2)
    dilated_img = cv2.erode(dilated_img, kernel, iterations=1)
    # filling the brick, will affect performance
    # brick_image = np.min(image, axis=2)
    # brick_image[brick_image>120] = 255
    # brick_image[brick_image<120] = 0
    # end = np.maximum(brick_image, dilated_img)


    # taking the side windows and calculate feature
    L_patch = np.copy(dilated_img[int(L_up_lim * image.shape[0]):int(L_down_lim * image.shape[0]), 60:210])
    # R_patch might won't work, 1000 is out of the image
    R_patch = np.copy(dilated_img[int(R_up_lim * image.shape[0]):int(R_down_lim * image.shape[0]), 450:600])

    Right_road_precent = (1 - np.sum(R_patch) / (R_patch.shape[0] * R_patch.shape[1] * 255)) * 100  # in percent!!!!!
    Left_road_precent = (1 - np.sum(L_patch) / (L_patch.shape[0] * L_patch.shape[1] * 255)) * 100

    temp2[:, :, 0] = dilated_img
    temp2[:, :, 1] = dilated_img
    temp2[:, :, 2] = dilated_img

    # decide if we about to go out
    '''
    if (Right_road_precent < precent_thresh or Left_road_precent < precent_thresh):
        temp[:, :, 2] = end
        temp2[:, :, 2] = dilated_img
        print('error')
        
    else:
        temp[:, :, 0] = end
        temp[:, :, 1] = end
        temp[:, :, 2] = end
        temp2[:, :, 0] = dilated_img
        temp2[:, :, 1] = dilated_img
        temp2[:, :, 2] = dilated_img
    '''
    # collision from the right
    if Right_road_precent < precent_thresh:
        stop_counter_R += 1

    stop_R = stop_counter_R == counter_thresh

    if stop_R:
        print("stop")
        #cv2.imshow('input', stop_image_R)
        temp2 = np.zeros_like(temp2)
        temp2[:, :, 2] = dilated_img
        out_bin.write(im)
        cv2.waitKey(1000)
        stop_L = False
        stop_counter_R = 0
        where_to_go()
        continue

    # collision from the left
    if Left_road_precent < precent_thresh:
        stop_counter_L += 1

    stop_L = stop_counter_L == counter_thresh

    if stop_L:
        print("stop")
        #cv2.imshow('input', stop_image_L)
        temp2 = np.zeros_like(temp2)
        temp2[:, :, 2] = dilated_img
        #out_bin.write(im)
        # cv2.waitKey(1000) #maybe we still need this
        stop_L = False
        stop_counter_L = 0
        where_to_go()
        continue

    '''
    save a picture from the camera with the interesting pats highlighted 
    image[int(L_up_lim*image.shape[0]):int(L_down_lim*image.shape[0]),60:210, np.array([1, 2])] = 0
    image[int(R_up_lim*image.shape[0]):int(R_down_lim*image.shape[0]), 1000:1150, np.array([1, 2])] = 0
    image[int(end_idx - 2*win_y):end_idx,  int(sat.shape[1]/2 - win_x):int(sat.shape[1]/2 + win_x), np.array([1, 2])] = 0
    cv2.imwrite('itamar.png', image)
    '''

    Hori = np.hstack((h_s_v[:, :, 0], h_s_v[:, :, 1], h_s_v[:, :, 2]))

    # uncomment to see red image when the robot is about to go out from the track (image after we "filled" the bricks )
    # cv2.imshow('input', temp)

    # uncomment to see dilated image turn red when the robot is about to go out from the track
    # cv2.imshow('input', temp2)

    # uncomment to see h, s and v side by side
    # cv2.imshow('input', Hori)

    # write image to video
    out.write(frame)
    out_bin.write(temp2)

    c = cv2.waitKey(20)
    i+=1

vid.release()
out.release()
out_bin.release()

cv2.destroyAllWindows()
