import cv2
import serial
import time
import numpy as np
from ultralytics import YOLO

# --- SETTINGS ---
SERIAL_PORT = '/dev/ttyACM0'
esp32 = serial.Serial(SERIAL_PORT, 921600, timeout=0)
model = YOLO("yolov8n-pose.pt")
cap = cv2.VideoCapture(0)

# PID Tuning (Applied to both axes)
pid = {'p': 3.5, 'i': 0.02, 'd': 0.15}
DEADZONE = 5
CENTER_X = 320
CENTER_Y = 240

# Acceleration Control
MAX_ACCEL = 5
actual_speed_x = 0.0
actual_speed_y = 0.0

error_sum_x = 0
last_error_x = 0
error_sum_y = 0
last_error_y = 0
last_time = time.time()

print("Dual-Axis Smooth PID Tracking Active. Press 'q' to stop.")

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)

    now = time.time()
    dt = now - last_time
    last_time = now

    results = model.predict(frame, verbose=False, conf=0.5)
    target_x = CENTER_X
    target_y = CENTER_Y
    found = False

    if results and results[0].keypoints:
        kp = results[0].keypoints.xy[0].cpu().numpy()
        # Check if nose is detected (coordinates > 0)
        if len(kp) > 0 and kp[0][0] > 0 and kp[0][1] > 0:
            target_x = int(kp[0][0])
            target_y = int(kp[0][1])
            found = True

    # ==========================================
    # --- X-AXIS (YAW) PID & RAMPING ---
    # ==========================================
    error_x = target_x - CENTER_X

    if found and abs(error_x) > DEADZONE:
        error_sum_x += error_x * dt
        derivative_x = (error_x - last_error_x) / dt if dt > 0 else 0
        output_x = (pid['p'] * error_x) + (pid['i'] * error_sum_x) + (pid['d'] * derivative_x)

        target_speed_x = np.clip(abs(output_x), 0, 255)
        dir_x = 1 if output_x > 0 else 2

        speed_diff_x = target_speed_x - actual_speed_x
        if abs(speed_diff_x) > MAX_ACCEL:
            actual_speed_x += np.sign(speed_diff_x) * MAX_ACCEL
        else:
            actual_speed_x = target_speed_x
    else:
        if actual_speed_x > 0:
            actual_speed_x -= MAX_ACCEL * 1.5
            if actual_speed_x < 0: actual_speed_x = 0
        dir_x = 1
        error_sum_x = 0

    last_error_x = error_x

    # ==========================================
    # --- Y-AXIS (PITCH) PID & RAMPING ---
    # ==========================================
    error_y = target_y - CENTER_Y

    if found and abs(error_y) > DEADZONE:
        error_sum_y += error_y * dt
        derivative_y = (error_y - last_error_y) / dt if dt > 0 else 0
        output_y = (pid['p'] * error_y) + (pid['i'] * error_sum_y) + (pid['d'] * derivative_y)

        target_speed_y = np.clip(abs(output_y), 0, 255)
        # Note: Pitch direction might need flipping (swap 1 and 2 here if it looks the wrong way)
        dir_y = 1 if output_y > 0 else 2

        speed_diff_y = target_speed_y - actual_speed_y
        if abs(speed_diff_y) > MAX_ACCEL:
            actual_speed_y += np.sign(speed_diff_y) * MAX_ACCEL
        else:
            actual_speed_y = target_speed_y
    else:
        if actual_speed_y > 0:
            actual_speed_y -= MAX_ACCEL * 1.5
            if actual_speed_y < 0: actual_speed_y = 0
        dir_y = 1
        error_sum_y = 0

    last_error_y = error_y

    # --- SEND COMMAND TO ESP32-C6 ---
    final_speed_x = int(actual_speed_x)
    final_speed_y = int(actual_speed_y)

    # Format: dirX,speedX,dirY,speedY\n
    esp32.write(f"{dir_x},{final_speed_x},{dir_y},{final_speed_y}\n".encode())

    # --- UI DISPLAY ---
    cv2.putText(frame, f"X SPD: {final_speed_x} | Y SPD: {final_speed_y}", (10, 30), 0, 0.7, (0, 255, 0), 2)

    # Draw X Deadzone
    cv2.line(frame, (CENTER_X - DEADZONE, 0), (CENTER_X - DEADZONE, 480), (0, 255, 255), 1)
    cv2.line(frame, (CENTER_X + DEADZONE, 0), (CENTER_X + DEADZONE, 480), (0, 255, 255), 1)
    # Draw Y Deadzone
    cv2.line(frame, (0, CENTER_Y - DEADZONE), (640, CENTER_Y - DEADZONE), (0, 255, 255), 1)
    cv2.line(frame, (0, CENTER_Y + DEADZONE), (640, CENTER_Y + DEADZONE), (0, 255, 255), 1)

    if found: cv2.circle(frame, (target_x, target_y), 10, (0, 255, 0), -1)

    cv2.imshow("Smoothed Dual-Axis Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
esp32.close()
