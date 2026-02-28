import cv2
import time
import math
import numpy as np
from mediapipe.tasks.python.vision import hand_landmarker
from mediapipe.tasks.python.vision.core import image as mp_image

# -----------------------------
# Finger gestures detection
# -----------------------------
def is_fist(hand):
    tips = [8, 12, 16, 20]  # index, middle, ring, pinky
    mcps = [5, 9, 13, 17]
    for tip, mcp in zip(tips, mcps):
        if hand[tip].y < hand[mcp].y:  # finger up
            return False
    return True

def is_index_finger_up(hand, margin=0.02):
    tip = hand[8]
    pip = hand[6]
    mcp = hand[5]
    other_tips = [12, 16, 20]  # middle, ring, pinky
    other_mcps = [9, 13, 17]
    index_extended = tip.y < pip.y - margin and pip.y < mcp.y - margin
    others_down = all(hand[t].y > hand[m].y + margin for t, m in zip(other_tips, other_mcps))
    return index_extended and others_down

def is_pinch(hand, threshold=0.1):
    thumb_tip = hand[4]
    index_tip = hand[8]
    distance = math.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)
    return distance < threshold

# -----------------------------
# Model setup
# -----------------------------
model_path = "hand_landmarker.task"
latest_result = None

def hand_result_callback(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

options = hand_landmarker.HandLandmarkerOptions(
    base_options=hand_landmarker._BaseOptions(model_asset_path=model_path),
    running_mode=hand_landmarker.HandLandmarkerOptions.running_mode.LIVE_STREAM,
    result_callback=hand_result_callback
)

landmarker = hand_landmarker.HandLandmarker.create_from_options(options)

# -----------------------------
# Canvas & viewport setup
# -----------------------------
CAMVAS_WIDTH = 3000
CANVAS_HEIGHT = 2000

canvas = np.zeros((CANVAS_HEIGHT, CAMVAS_WIDTH, 3), dtype=np.uint8)
canvas_offset = [0, 0]  # x, y offset of the viewport
prev_canvas_point = None
pinch_start = None

# -----------------------------
# Webcam setup
# -----------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

start_time = time.time()
_, frame = cap.read()
frame_h, frame_w = frame.shape[:2]

# -----------------------------
# Hand connections for drawing
# -----------------------------
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]

# -----------------------------
# Main loop
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_frame = mp_image.Image(image_format=mp_image.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms = int((time.time() - start_time) * 1000)
    landmarker.detect_async(mp_frame, timestamp_ms)

    # Only draw if we have results
    if latest_result and latest_result.hand_landmarks:
        hand = latest_result.hand_landmarks[0]  # Only handle first hand

        # -----------------
        # Drawing with pointer
        # -----------------
        if is_index_finger_up(hand):
            x = int(hand[8].x * frame_w) + canvas_offset[0]
            y = int(hand[8].y * frame_h) + canvas_offset[1]
            if prev_canvas_point is not None:
                cv2.line(canvas, prev_canvas_point, (x, y), (0, 255, 0), 10)
            prev_canvas_point = (x, y)
            pinch_start = None  # cancel pan if drawing

        # -----------------
        # Pan with pinch
        # -----------------
        elif is_pinch(hand):
            pinch_x = int(hand[8].x * frame_w)
            pinch_y = int(hand[8].y * frame_h)
            if pinch_start is None:
                pinch_start = (pinch_x, pinch_y)
            else:
                dx = pinch_x - pinch_start[0]
                dy = pinch_y - pinch_start[1]
                canvas_offset[0] = max(0, min(CAMVAS_WIDTH - frame_w, canvas_offset[0] - dx))
                canvas_offset[1] = max(0, min(CANVAS_HEIGHT - frame_h, canvas_offset[1] - dy))
                pinch_start = (pinch_x, pinch_y)
            prev_canvas_point = None  # cancel drawing if panning


            # fist
        elif is_fist(hand):
            canvas = np.zeros((CANVAS_HEIGHT, CAMVAS_WIDTH, 3), dtype=np.uint8)

        else:
            prev_canvas_point = None
            pinch_start = None

        # -----------------
        # Draw hand landmarks on frame
        # -----------------
        for landmark in hand:
            lx = int(landmark.x * frame_w)
            ly = int(landmark.y * frame_h)
            cv2.circle(frame, (lx, ly), 5, (0, 0, 255), -1)

        for start_idx, end_idx in HAND_CONNECTIONS:
            start = hand[start_idx]
            end = hand[end_idx]
            x1, y1 = int(start.x * frame_w), int(start.y * frame_h)
            x2, y2 = int(end.x * frame_w), int(end.y * frame_h)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # -----------------
    # Show viewport of canvas on frame
    # -----------------
    x_off, y_off = canvas_offset
    viewport = canvas[y_off:y_off+frame_h, x_off:x_off+frame_w]
    display = cv2.add(frame, viewport)
    cv2.imshow("Pointer & Pinch Drawing", display)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()