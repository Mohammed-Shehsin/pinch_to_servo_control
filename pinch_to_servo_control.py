import cv2
import math
from collections import deque
import numpy as np
import mediapipe as mp

# -------------------- Parameters --------------------
SMOOTH_N = 7                   # moving-average window for angle smoothing
PINCH_FORCE_ZERO = 0.08        # if normalized r < this → force angle = 0
R_MIN_DEFAULT = 0.10           # normalized pinch at (almost) touch
R_MAX_DEFAULT = 0.60           # normalized pinch at comfortably spread
ANGLE_MIN, ANGLE_MAX = 0, 180
DRAW_SCALE = 1                 # UI scale factor

# -------------------- Setup MediaPipe --------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera 0")

# -------------------- State --------------------
r_min, r_max = R_MIN_DEFAULT, R_MAX_DEFAULT
angle_hist = deque(maxlen=SMOOTH_N)

def l2(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def map_norm_to_angle(r, rmin, rmax):
    if r <= PINCH_FORCE_ZERO:
        return 0.0
    t = (r - rmin) / max(1e-6, (rmax - rmin))
    t = clamp(t, 0.0, 1.0)
    return ANGLE_MIN + t * (ANGLE_MAX - ANGLE_MIN)

def draw_bar(img, angle, x=40, y=40, h=220, w=20):
    # Outline
    cv2.rectangle(img, (x, y), (x+w, y+h), (255,255,255), 2)
    # Fill
    fill_h = int((angle/180.0) * (h-4))
    cv2.rectangle(img, (x+2, y+h-2-fill_h), (x+w-2, y+h-2), (255,255,255), -1)
    # Ticks
    for k in [0, 45, 90, 135, 180]:
        ty = y + h - int((k/180.0) * (h-4)) - 2
        cv2.line(img, (x+w+6, ty), (x+w+26, ty), (255,255,255), 1)
        cv2.putText(img, f"{k}", (x+w+30, ty+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

def put_text(img, text, org, scale=0.8, thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thickness+2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), thickness, cv2.LINE_AA)

# -------------------- Main loop --------------------
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)                    # mirror view
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        angle_out = None
        if res.multi_hand_landmarks:
            handLms = res.multi_hand_landmarks[0]
            # Landmarks of interest
            lm = handLms.landmark
            # Thumb tip (4), Index tip (8), Index MCP (5), Pinky MCP (17)
            pts = {}
            for idx in [4, 8, 5, 17]:
                x = int(lm[idx].x * w)
                y = int(lm[idx].y * h)
                pts[idx] = (x, y)

            # Distances in pixels
            d_tip = l2(pts[4], pts[8])
            d_ref = max(10.0, l2(pts[5], pts[17]))  # avoid divide-by-zero; use hand width as scale

            r = d_tip / d_ref  # normalized pinch

            # Map to angle
            angle = map_norm_to_angle(r, r_min, r_max)
            angle_hist.append(angle)
            angle_smooth = float(np.mean(angle_hist))
            angle_out = angle_smooth

            # ----- Draw UI -----
            # skeleton
            mp_drawing.draw_landmarks(
                frame, handLms, mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )
            # highlight thumb & index tips
            cv2.circle(frame, pts[4], 10, (0,255,255), -1)
            cv2.circle(frame, pts[8], 10, (0,255,255), -1)
            cv2.line(frame, pts[4], pts[8], (255,255,255), 2)

            # reference width line
            cv2.circle(frame, pts[5], 6, (255,255,255), -1)
            cv2.circle(frame, pts[17], 6, (255,255,255), -1)
            cv2.line(frame, pts[5], pts[17], (200,200,200), 1)

            # text overlays
            put_text(frame, f"r = d_tip/d_ref = {r:.3f}", (40, 300))
            put_text(frame, f"r_min={r_min:.2f}  r_max={r_max:.2f}", (40, 330))
            put_text(frame, f"Angle = {int(round(angle_smooth))} deg", (40, 270), scale=1.0)

            # bar
            draw_bar(frame, angle_smooth, x=40, y=40, h=220, w=22)

        else:
            put_text(frame, "Show one hand to the camera", (40, 80))

        # help text
        put_text(frame, "Keys: Z=set ZERO | X=set MAX | Q=quit", (40, h-30), scale=0.7, thickness=1)

        cv2.imshow("Pinch-to-Servo (0..180)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('z'):
            # Set current normalized r as new r_min (zero)
            if res.multi_hand_landmarks and angle_out is not None:
                r_min = max(0.0, min(r, r_max - 0.02))  # keep r_min < r_max
        elif key == ord('x'):
            # Set current normalized r as new r_max
            if res.multi_hand_landmarks and angle_out is not None:
                r_max = max(r_min + 0.02, r)

finally:
    cap.release()
    hands.close()
    cv2.destroyAllWindows()
