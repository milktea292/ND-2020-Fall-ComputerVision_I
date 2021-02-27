# Computer Vision Course (CSE 40535/60535)
# University of Notre Dame, Fall 2019
# ______________________________________________________________________
# Jin Huang, Aakash Bansal, Adam Czajka, September 2018 -  November 2020

import cv2
import numpy as np
from skimage import measure
from sys import platform as sys_pf
import warnings
warnings.filterwarnings("ignore")
from numpy.linalg import multi_dot
from numpy.linalg import lstsq

if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
plt.plot()

font = cv2.FONT_HERSHEY_SIMPLEX

# Parameters and paths
video_path = "MarkerCap.mp4"

# Note 1: for high FPS (frame per second), exceeding 25, consider skipping
# every k frames (i.e., make "frameStep" larger than 1):
frame_step = 3

# HSV threshold for the object
h_lower, h_upper = 63, 75
s_lower, s_upper = 25, 255
v_lower, v_upper = 51, 230

# Kalman filter parameters
kal_x = np.matrix([0, 0, 0, 0]) # state, in this case rigged as [x y dx dy]
kal_P = np.identity(4) # filter confidence (state covariance)
kal_C = np.matrix([[1, 0, 0, 0],
                   [0, 1, 0, 0]])

# Transition matrix
dt = 1
kal_A = np.matrix([[1, 0, dt, 0],
                    [0, 1,  0, dt],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

# Noise estimates
kal_R = 0.1 * np.identity(2) # measurement noise
kal_Q = 0.1 * np.identity(4) # process noise


# Definition of "Lego bricks" for this practical:

def object_detect(frame):
    image_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # # Select the proper hsv value
    # mask = np.zeros((image_hsv.shape[0], image_hsv.shape[1]))

    # for i in range(image_hsv.shape[0]):
    #     for j in range(image_hsv.shape[1]):
    #         pixel_h = image_hsv[i, j, 0]
    #         pixel_s = image_hsv[i, j, 1]
    #         pixel_v = image_hsv[i, j, 2]

    #         if ((pixel_h >= h_lower) and (pixel_h <= h_upper) and
    #             (pixel_s >= s_lower) and (pixel_s <= s_upper) and
    #             (pixel_v >= v_lower) and (pixel_v <= v_upper)):

    #             mask[i, j] = 1

    lower = np.array([63, 25, 51])
    upper = np.array([75, 125, 230])
    mask = cv2.inRange(image_hsv, lower, upper)

    labels = measure.label(mask, 4)
    features = measure.regionprops(labels)

    if (features != None):
        # Find the index of the biggest blob
        all_area_size = []

        for i in range(len(features)):
            area_size = features[i].area
            all_area_size.append(area_size)

        # Find the index of the biggest area
        index_of_biggest_blob = np.argmax(all_area_size)

        # Get the coordinates of the centroid
        x = features[index_of_biggest_blob].centroid[0]
        y = features[index_of_biggest_blob].centroid[1]

        # position = [x, y]
        position = [[x], [y]]

        return position

def kalman_predict(A, X):
    return np.dot(A, X.getH())

def kalman_update(kal_P_init, predicted, measured):
    Pk = multi_dot([kal_A, kal_P_init, kal_A.getH()]) + kal_Q

    K_left_matrix = np.dot(Pk, kal_C.getH())
    K_right_matrix = multi_dot([kal_C, Pk, kal_C.getH()]) + kal_R
    K = lstsq(K_left_matrix.T, K_right_matrix.T)[0]
    kal_x = (np.matrix(predicted) +
             np.dot(K, (np.matrix(measured) -
                        np.dot(kal_C, np.matrix(predicted))))).getH()
    kal_P = np.dot((np.identity(4) - np.dot(K, kal_C)), Pk)

    return Pk, K, kal_x, kal_P


if __name__ == '__main__':

    # Load the video file and count number of frames
    # cap = cv2.VideoCapture(video_path)
    # frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # print("There are %d frames in the video." % frame_count)

    cam = cv2.VideoCapture(0)
    nb_frame = 0
    print("Tracking started...")

    # while (cap.isOpened() and (nb_frame < frame_count - frame_step)):
        # for i in range(frame_step):
        #     ret, frame_current = cap.read()
        #     nb_frame += 1
    while(True):
        retval, frame_current = cam.read()

        # Measurement, prediction and update

        # ***Task for you*** (0.5 points)
        # Make a Kalman prediction here: 
        predicted_state = kalman_predict(kal_A, kal_x)

        # Object detection (measurement)
        try:
            measured_position = object_detect(frame_current)

            center_coordinates = (int(np.round(measured_position[1])),
                                  int(np.round(measured_position[0])))

            cv2.circle(frame_current, center_coordinates, 10, (0, 255, 0), 2)
            center_coordinates_label = tuple(map(sum, zip(center_coordinates, (20,0))))
            cv2.putText(frame_current,    # image
                'detected',  # text
                center_coordinates_label,    # start position
                font,       # font
                0.5,        # size
                (0, 255, 0),# BGR color
                1,          # thickness
                cv2.LINE_AA) # type of line

        except ValueError:
            
            # ***Task for you*** (1 point)
            # If we do not have a measurement for this frame, let's use the _predicted_ state:
            # measured_position = np.zeros(2) # change this!
            measured_position = predicted_state[0:2]

        # ***Task for you*** (0.5 points)
        # Make the Kalman update here with kal_P_init = kal_P, predicted = predicted_state and measured = measured_position
        Pk, K, kal_x, kal_P = kalman_update(kal_P_init = kal_P,
                                            predicted  = predicted_state,
                                            measured   = measured_position)

        # ***Task for you*** (0.5 points)
        # Where is the estimated position?
        estimated_position = np.zeros(2) # change this! For instance:
        estimated_position[0] = np.array(predicted_state[0])
        estimated_position[1] = np.array(predicted_state[1])
        
        center_coordinates_estimate = (int(np.round(estimated_position[1])),
                                        int(np.round(estimated_position[0])))

        cv2.circle(frame_current, center_coordinates_estimate, 10, (255, 0, 0), 2)
        center_coordinates_estimate_label = tuple(map(sum, zip(center_coordinates_estimate, (20,15))))
        cv2.putText(frame_current,                      # image
                'Kalman estimation',                    # text
                center_coordinates_estimate_label,      # start position
                font,                                   # font
                0.5,                                    # size
                (255, 0, 0),                            # BGR color
                1,                                      # thickness
                cv2.LINE_AA)                            # type of line

        cv2.imshow("Tracking result", frame_current)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    print("Tracking finished.")
    #cap.release()
    cv2.destroyAllWindows()


'''
***Task for you*** (1.5 points):
Change this code to work with a webcam stream, 
as in the first practical, for a single color object.
Save it as kalman_1_sensor_webcam.py in SAKAI Dropbox.
'''