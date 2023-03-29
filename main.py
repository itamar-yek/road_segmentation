import numpy as np
import cv2
import helper as h


if __name__ == "__main__":
    ''''
    the main function for road segmantation 
    if you want to save the output video and the segmantation, set save_vid = True, the desired path in saved_path annd the name.
    the inst variable need to be adjusted in order to send commands to the robot.
    '''

    save_vid = False
    saved_path = ''
    saved_name = 'vid_exp'
    cam_feed = 'vid3.mp4' #0 or 1 for live feed (depend on how many cameras are connected to the computer)
    precent_thresh = 40
    stop_R = False
    stop_L = False
    stop_counter_R = 0
    stop_counter_L = 0
    counter_thresh = 30
    ret = True

    i=0
    
    vid = cv2.VideoCapture(cam_feed)
    if save_vid:      
        # saving video
        ret, frame = vid.read()
        fps = 20
        vid_format = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(saved_path + saved_name + '.mp4', vid_format, fps, (frame.shape[1] // 2, frame.shape[0]))
        out_bin = cv2.VideoWriter(saved_path + saved_name + '_bin.mp4', vid_format, fps, (frame.shape[1]//2, frame.shape[0]), 0)
    
    while ret:
        ret, frame = vid.read()
        # depend on the camera
        frame = np.copy(frame[:, frame.shape[1] // 2:, :])  # take only the left camera
        segment = h.convert2bin(frame)
        left_percent, right_percent = h.side_patches(segment)

        stop_counter_L += (left_percent < precent_thresh)
        stop_counter_R += (right_percent < precent_thresh)

        stop_R = stop_counter_R == counter_thresh
        stop_L = stop_counter_L == counter_thresh
        stop = stop_R + stop_L 

        inst = "straight"
        if stop > 0:
            stop_counter_L = stop_counter_R = 0
            inst = h.where2go(vid, stop_R, stop_L)
            print(str(inst))
        
        temp = np.copy(segment[frame.shape[0] // 2:, :])
        frame_blue = np.copy(frame[frame.shape[0] // 2:, :, 1])
        frame_blue[temp == 0] = 0
        frame[frame.shape[0] // 2:, :, 1] = frame_blue


        cv2.imwrite("gif_frames/image"+str(i)+".png", frame)
        i+=1


        if save_vid:
            out.write(frame)
            out_bin.write(segment)
