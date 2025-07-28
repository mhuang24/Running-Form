from os import supports_dir_fd

import cv2
import mediapipe as mp
import numpy as np
import math

from mediapipe.python.solutions.drawing_utils import RED_COLOR

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic

#VIDEO FEED
cap = cv2.VideoCapture("IMG_5396.mp4")
paused = False
total_height = 0
frame_counter = 0
image_width = 540
image_height = 960

right_heel_y=[]
right_toe_y=[]
right_heel_acc = []
right_toe_acc = []

MIN_VISIBILITY = 0.7
ACC_THRESH = -0.003
MIN_CYCLES = 5

last_strike = 0
strike_frames = []


def calculate_angle(a, b, c):
    a = np.array(a) #First landmark
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0],) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle %360

def angle_between_vectors(v1, v2):
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_v1, unit_v2)
    angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
    return np.degrees(angle_rad)


with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.99) as pose:
    while cap.isOpened():
        if not paused:
            frame_counter += 1
            ret, frame = cap.read()
            #frame = cv2.resize(frame, (960, 540))
            frame = cv2.resize(frame, (image_width, image_height))
            #Detect stuff and render
            #Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            #Make detection
            results = pose.process(image)
            #Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            #Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                """
                Y_DIFF_THRESH = 0.015  # depends on resolution, tune this

                toe_y = landmarks[32].y
                heel_y = landmarks[30].y
                foot_y_diff = abs(toe_y - heel_y)

                # Optional: check foot is also near the ground (e.g., bottom 20% of image)
                LOW_Y_THRESH = 0.6

                if (foot_y_diff == 0) and (toe_y > LOW_Y_THRESH) and (heel_y > LOW_Y_THRESH):
                    # Potential foot strike
                    paused = not paused
                """
                """
                right_toe_vis = landmarks[32].visibility
                right_heel_vis = landmarks[30].visibility
                if right_toe_vis < MIN_VISIBILITY or right_heel_vis < MIN_VISIBILITY:
                    continue

                right_toe_y.append(landmarks[32].y)
                right_heel_y.append(landmarks[30].y)

                if len(right_heel_y) < 4:
                    continue

                heel_acc = right_heel_y[-1] - 2 * right_heel_y[-2] + right_heel_y[-3]
                toe_acc = right_toe_y[-1] - 2 * right_toe_y[-2] + right_toe_y[-3]

                is_low_point = (right_heel_y[-1] > right_heel_y[-2]) and (right_heel_y[-1] > right_heel_y[-3]) and (right_toe_y[-1] > right_toe_y[-2] and right_toe_y[-1] > right_toe_y[-3])

                if (heel_acc > ACC_THRESH and toe_acc > ACC_THRESH) and is_low_point:
                    paused = not paused
                    #if not strike_frames or (frame_counter - strike_frames[-1] > MIN_CYCLES):
                    #    strike_frames.append(frame_counter)
                    #    paused = not paused
                """
                # Update running average of foot to shoulder height
                # total_height += math.sqrt(abs(landmarks[12].x - landmarks[24].x) ** 2 + abs(landmarks[12].y - landmarks[24].y) ** 2) + math.sqrt(
                #                           abs(landmarks[24].x - landmarks[26].x) ** 2 + abs(landmarks[24].y - landmarks[26].y) ** 2) + math.sqrt(
                #                           abs(landmarks[28].x - landmarks[26].x) ** 2 + abs(landmarks[28].y - landmarks[26].y) ** 2)
                # avg_height = total_height / frame_counter


                #Right knee angle
                right_hip = (landmarks[24].x, landmarks[24].y)
                right_knee = (landmarks[26].x, landmarks[26].y)
                right_ankle = (landmarks[28].x, landmarks[28].y)
                knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
                right_knee_coords = tuple(np.multiply(
                    [right_knee[0], right_knee[1]],
                    [image_width, image_height]
                ).astype(int))
                cv2.putText(image, str(math.trunc(knee_angle)),
                            (right_knee_coords[0] + 10, right_knee_coords[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )

                #Right hip angle
                right_shoulder = (landmarks[12].x, landmarks[12].y)
                right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
                right_hip_coords = tuple(np.multiply(
                    [right_hip[0], right_hip[1]],
                    [image_width, image_height]
                ).astype(int))
                cv2.putText(image, str(math.trunc(right_hip_angle)),
                            (right_hip_coords[0] + 10, right_hip_coords[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                #Right ankle angle
                right_knee = landmarks[26]
                right_ankle = landmarks[28]
                right_toe = landmarks[32]
                right_heel = landmarks[30]

                shin_vec = np.array([right_ankle.x - right_knee.x, right_ankle.y - right_knee.y])
                foot_vec = np.array([right_toe.x - right_heel.x, right_toe.y - right_heel.y])

                right_ankle_angle = angle_between_vectors(shin_vec, foot_vec)
                right_ankle_coords = tuple(np.multiply(
                    [right_ankle.x, right_ankle.y],
                    [image_width, image_height]
                ).astype(int))

                cv2.putText(image, str(math.trunc(right_ankle_angle)),
                            (right_ankle_coords[0] + 10, right_ankle_coords[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                right_toe_vis = landmarks[32].visibility
                right_heel_vis = landmarks[30].visibility
                if right_toe_vis < MIN_VISIBILITY or right_heel_vis < MIN_VISIBILITY:
                    continue

                right_toe_y.append(landmarks[32].y)
                right_heel_y.append(landmarks[30].y)

                if len(right_heel_y) < 4:
                    continue

                heel_acc = right_heel_y[-1] - 2 * right_heel_y[-2] + right_heel_y[-3]
                toe_acc = right_toe_y[-1] - 2 * right_toe_y[-2] + right_toe_y[-3]
                cv2.putText(image, "toe y: "+str(round(landmarks[32].y, 4)), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, "heel y: "+str(round(landmarks[30].y, 4)), (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, "toe_acc: "+str(round(toe_acc, 4)), (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, "heel_acc: "+str(round(heel_acc, 4)), (10,110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)


                is_low_point = (right_heel_y[-2] < right_heel_y[-1]) and (right_heel_y[-2] < right_heel_y[-3]) and (
                        right_toe_y[-2] < right_toe_y[-1] and right_toe_y[-2] < right_toe_y[-3])

                #Check for local maximum in y(lowest point) and near-zero vertical velocity
                #if abs(right_ankle.x - right_knee.x) < .01:
                if (toe_acc < 0 and heel_acc < 0 and abs(landmarks[32].y - landmarks[30].y) < .01):
                    #Set timer for cycles
                    temp = frame_counter - last_strike
                    if temp > 30:
                        paused = not paused
                        last_strike = frame_counter
                        strike_frames = []
                    else:
                        strike_frames.append(frame_counter)

                if frame_counter - last_strike > 30:
                    paused = not paused
                    max_x = 0
                    for i in strike_frames:
                        max_x = max(max_x, i.x)
                        print(max_x)
                    #go to frame with max x


                    # if (heel_acc > ACC_THRESH and toe_acc > ACC_THRESH) and is_low_point:
                    #     temp = frame_counter - last_strike
                    # if temp > 20:
                    #     last_strike = frame_counter
                    #paused = not paused
                        # if not strike_frames or (frame_counter - strike_frames[-1] > MIN_CYCLES):
                        #    strike_frames.append(frame_counter)
                        #    paused = not paused




            except:
                pass

            #Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(255,0,0),thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(0,0,255),thickness=2, circle_radius=2),)

            cv2.imshow("Frame", image)

            #print(results)

            #cv2.imshow('MediaPipe Feed', frame)
        key = cv2.waitKey(1) & 0xFF  # 30ms delay between frames

        if key == ord('p'):
            paused = not paused  # Toggle pause

        elif key == ord('q') or key == 27:
            break  # Quit on 'q' or ESC

    cap.release()
    cv2.destroyAllWindows()

