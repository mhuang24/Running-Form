import cv2
import mediapipe as mp
import numpy as np
from collections import deque

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# --- Config ---
WORK_W, WORK_H = 540, 960
MIN_VISIBILITY = 0.7
Y_DIFF_FLAT_THRESH = 0.015  # toe vs heel y should be close (normalized)
DEBOUNCE_SEC = 0.25  # minimum time between strikes
PAST_WINDOW_FRAMES = 30  # choose best frame from recent window


# --- Helpers ---
def calc_angle(a, b, c):
    a = np.array(a, float);
    b = np.array(b, float);
    c = np.array(c, float)
    ab = a - b;
    cb = c - b
    cosang = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb) + 1e-9)
    return np.degrees(np.arccos(np.clip(cosang, -1, 1)))


def angle_between(v1, v2):
    v1 = np.array(v1, float);
    v2 = np.array(v2, float)
    v1 /= (np.linalg.norm(v1) + 1e-9);
    v2 /= (np.linalg.norm(v2) + 1e-9)
    return np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1, 1)))


def to_px(pt, W, H): return (int(pt[0] * W), int(pt[1] * H))


# Finite differences
def first_diff(series):
    if len(series) < 2: return None
    return series[-1] - series[-2]


def velocity_zero_crossing(v_series):
    # detect last crossing from negative -> non-negative
    if len(v_series) < 2: return False
    return (v_series[-2] > 0) and (v_series[-1] <= 0)


# --- Init ---
cap = cv2.VideoCapture("IMG_5396.mp4")
if not cap.isOpened():
    raise RuntimeError("Could not open video")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
debounce_frames = int(DEBOUNCE_SEC * fps)

paused = False
frame_idx = -1
last_strike_frame = -10 ** 9
next_strike_frame = None
skip_display_this_iteration = False
pause_after_next_frame = False

# Rolling series for y and vy
toe_y = deque(maxlen=256)
heel_y = deque(maxlen=256)
toe_v = deque(maxlen=256)
heel_v = deque(maxlen=256)

# Rolling recent frame buffer: store (frame_index, toe_x)
recent_frames = deque(maxlen=90)

with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.99) as pose:
    while True:
        skip_display_this_iteration = False

        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            # Resize to your working resolution
            frame = cv2.resize(frame, (WORK_W, WORK_H))
            H, W = frame.shape[:2]

            # Mediapipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                R_SH = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
                R_HI = mp_pose.PoseLandmark.RIGHT_HIP.value
                R_KN = mp_pose.PoseLandmark.RIGHT_KNEE.value
                R_AN = mp_pose.PoseLandmark.RIGHT_ANKLE.value
                R_HE = mp_pose.PoseLandmark.RIGHT_HEEL.value
                R_TO = mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value

                if lm[R_TO].visibility >= MIN_VISIBILITY and lm[R_HE].visibility >= MIN_VISIBILITY:
                    r_sh = (lm[R_SH].x, lm[R_SH].y)
                    r_hip = (lm[R_HI].x, lm[R_HI].y)
                    r_knee = (lm[R_KN].x, lm[R_KN].y)
                    r_ank = (lm[R_AN].x, lm[R_AN].y)
                    r_toe = (lm[R_TO].x, lm[R_TO].y)
                    r_heel = (lm[R_HE].x, lm[R_HE].y)

                    # angles (optional HUD)
                    knee_ang = calc_angle(r_hip, r_knee, r_ank)
                    hip_ang = calc_angle(r_sh, r_hip, r_knee)
                    shin_vec = (lm[R_AN].x - lm[R_KN].x, lm[R_AN].y - lm[R_KN].y)
                    foot_vec = (lm[R_TO].x - lm[R_HE].x, lm[R_TO].y - lm[R_HE].y)
                    ankle_ang = angle_between(shin_vec, foot_vec)
                    cv2.putText(image, f"{int(round(knee_ang))}",
                                (to_px(r_knee, W, H)[0] + 10, to_px(r_knee, W, H)[1] - 10), 0, 0.5, (255, 255, 255), 2)
                    cv2.putText(image, f"{int(round(hip_ang))}",
                                (to_px(r_hip, W, H)[0] + 10, to_px(r_hip, W, H)[1] - 10), 0, 0.5, (255, 255, 255), 2)
                    cv2.putText(image, f"{int(round(ankle_ang))}",
                                (to_px(r_ank, W, H)[0] + 10, to_px(r_ank, W, H)[1] - 10), 0, 0.5, (255, 255, 255), 2)

                    # update series & velocities (normalized coords)
                    toe_y.append(r_toe[1]);
                    heel_y.append(r_heel[1])
                    toe_v.append(first_diff(toe_y) or 0.0)
                    heel_v.append(first_diff(heel_y) or 0.0)

                    # keep recent frames buffer
                    recent_frames.append((frame_idx, r_toe[0]))

                    # Heuristic: velocity zero crossing (down -> up) + flat foot + debounce
                    strike = (
                            velocity_zero_crossing(toe_v) and
                            velocity_zero_crossing(heel_v) and
                            abs(r_toe[1] - r_heel[1]) < Y_DIFF_FLAT_THRESH and
                            (frame_idx - last_strike_frame) > debounce_frames
                    )

                    if strike:
                        last_strike_frame = frame_idx

                        # choose best frame from recent window (max toe x)
                        window = [t for t in recent_frames if frame_idx - t[0] <= PAST_WINDOW_FRAMES]
                        if window:
                            best_frame, _ = max(window, key=lambda t: t[1])
                        else:
                            best_frame = frame_idx

                        # seek & pause immediately
                        cap.set(cv2.CAP_PROP_POS_FRAMES, max(best_frame, 0))
                        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1  # next loop increments to exact frame
                        next_strike_frame = best_frame
                        pause_after_next_frame = True
                        skip_display_this_iteration = True

                # draw skeleton
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                )

            # Draw strike indicator if this is the strike frame
            if frame_idx == next_strike_frame:
                if results.pose_landmarks:
                    lm = results.pose_landmarks.landmark
                    R_TO = mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value
                    r_toe = (lm[R_TO].x, lm[R_TO].y)
                    cv2.circle(image, to_px(r_toe, W, H), 8, (0, 255, 0), -1)
                    cv2.putText(image, "STRIKE", (20, 140), 0, 0.8, (0, 255, 0), 2)
                    next_strike_frame = None

            # HUD
            cv2.putText(image, f"frame: {frame_idx}", (20, 30), 0, 0.6, (255, 255, 255), 2)

        # Display if not skipped
        if not paused and not skip_display_this_iteration:
            cv2.imshow("Frame", image)

        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('p'):
            paused = not paused
        elif key == ord('q') or key == 27:
            break
        elif paused and key == ord('n'):
            # single-step one frame when paused
            paused = False

        # Pause after next frame if requested
        if pause_after_next_frame:
            paused = True
            pause_after_next_frame = False

cap.release()
cv2.destroyAllWindows()