import cv2
import serial
import time
import numpy as np
import threading
import queue
import pyttsx3
from ultralytics import YOLO

cv2.setNumThreads(8)

SERIAL_PORT = '/dev/ttyACM0'  

GEAR_RATIO_X = 166.0 / 24.0 
GEAR_RATIO_Y = 60.0 / 20.0

FOV_X = 55.0
FOV_Y = 43.0

STEPS_PER_REV = 400.0
DEG_PER_STEP = 360.0 / STEPS_PER_REV  
STEP_DELAY_MS = 0.6 

INVERT_X = True
INVERT_Y = False

CENTER_X, CENTER_Y = 320, 240 
DEADZONE_PX = 8  

target_class = "person" 

abs_pos_x = 0 
abs_pos_y = 0 

print("Loading YOLO11 Models Locally (Auto-Detecting GPU/CPU)...")

model_pose = YOLO("yolo11n-pose.pt") 
model_obj = YOLO("yolo11x.pt")       

audio_queue = queue.Queue()
def audio_worker():
    engine = None
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 180)
    except: 
        print("[ERROR] TTS Engine failed to initialize")
        pass

    while True:
        m, content = audio_queue.get()
        if engine:
            if m == 'ding': 
                print("\a") 
                engine.say("Locked")
            elif m == 'desc':
                name, conf = content
                conf_pct = int(conf * 100)
                engine.say(f"I see a {name} with {conf_pct} percent confidence")
            engine.runAndWait()
        audio_queue.task_done()

threading.Thread(target=audio_worker, daemon=True).start()

class CameraStream:
    def __init__(self, index=2): 

        self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3) 

        actual_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"\n[CAMERA] Native Format: {actual_w}x{actual_h} @ {actual_fps} FPS\n")

        self.ret, self.frame = False, None
        self.started = False
        self.read_lock = threading.Lock()
        self.new_frame_event = threading.Event()

    def start(self):
        if self.started: return self
        self.started = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()
        return self

    def update(self):
        while self.started:
            self.cap.grab()
            ret, frame = self.cap.retrieve()
            if ret:
                with self.read_lock:
                    self.ret, self.frame = ret, frame
                self.new_frame_event.set()
            else:
                time.sleep(0.001)

    def read(self):
        if not self.new_frame_event.wait(timeout=0.1):
            with self.read_lock: return self.ret, self.frame
        self.new_frame_event.clear()
        with self.read_lock:
            return self.ret, self.frame

    def stop(self):
        self.started = False
        self.thread.join()
        self.cap.release()

def input_thread():
    global target_class
    while True:
        t = input().strip().lower()
        if t: target_class = t
threading.Thread(target=input_thread, daemon=True).start()

esp32 = None
try:
    esp32 = serial.Serial(SERIAL_PORT, 921600, timeout=0)
    print(f"ESP32 Connected on {SERIAL_PORT}")
except:
    print(f"ESP32 Not Connected on {SERIAL_PORT}. Continuing in simulation mode.")

cam = CameraStream(index=2).start()
last_time = time.time()
last_tts_time = 0
fps_smoothing = 0.9
current_fps = 60.0

while True:
    loop_start = time.time()
    ret, frame_raw = cam.read()
    if not ret or frame_raw is None: continue

    now = time.time()
    dt = now - last_time
    last_time = now

    tx, ty, found = CENTER_X, CENTER_Y, False
    dir_x, dir_y = 1, 1 
    steps_x, steps_y = 0, 0
    boxes_to_draw = []

    target_list = [t.strip() for t in target_class.split(',')]
    is_pose_mode = "person" in target_list

    if is_pose_mode:

        results = model_pose.predict(frame_raw, verbose=False, conf=0.45)
        if results and results[0].keypoints:
            kps = results[0].keypoints.xy.cpu().numpy()
            confs = results[0].keypoints.conf.cpu().numpy() if results[0].keypoints.conf is not None else [0.9] * len(kps)
            for i, person in enumerate(kps):
                if len(person) > 0 and person[0][0] > 0:
                    px, py = int(person[0][0]), int(person[0][1])
                    if not found:
                        tx, ty = px, py
                        found = True
                    conf = confs[i][0] if hasattr(confs[i], "__len__") else 0.9
                    boxes_to_draw.append(("person", [px-50, py-50, px+50, py+50], conf))
    else:

        results = model_obj.predict(frame_raw, verbose=False, conf=0.45)
        if results and results[0].boxes:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                cls_name = model_obj.names[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

                boxes_to_draw.append((cls_name, [x1, y1, x2, y2], conf))

                if cls_name in target_list and not found:
                    tx = int((x1 + x2) / 2)
                    ty = int((y1 + y2) / 2)
                    found = True

    if found:
        err_x_px = tx - CENTER_X
        err_y_px = ty - CENTER_Y

        if abs(err_x_px) < DEADZONE_PX: err_x_px = 0
        if abs(err_y_px) < DEADZONE_PX: err_y_px = 0

        err_x_deg = err_x_px * (FOV_X / 640.0)
        err_y_deg = err_y_px * (FOV_Y / 480.0)

        motor_x_deg = err_x_deg * GEAR_RATIO_X
        motor_y_deg = err_y_deg * GEAR_RATIO_Y

        steps_x = abs(int(motor_x_deg / DEG_PER_STEP))
        steps_y = abs(int(motor_y_deg / DEG_PER_STEP))

        if not INVERT_X: dir_x = 1 if err_x_px > 0 else 2
        else: dir_x = 2 if err_x_px > 0 else 1

        if not INVERT_Y: dir_y = 1 if err_y_px > 0 else 2
        else: dir_y = 2 if err_y_px > 0 else 1

        if time.time() - last_tts_time > 6.0 and len(boxes_to_draw) > 0:
            for name, coords, conf in boxes_to_draw:
                if name in target_list:
                    audio_queue.put(('desc', (name, conf)))
                    last_tts_time = time.time()
                    break
    else:

        steps_x = abs(abs_pos_x)
        steps_y = abs(abs_pos_y)

        steps_x = min(steps_x, 150)
        steps_y = min(steps_y, 150)

        if not INVERT_X: dir_x = 2 if abs_pos_x > 0 else 1
        else: dir_x = 1 if abs_pos_x > 0 else 2

        if not INVERT_Y: dir_y = 2 if abs_pos_y > 0 else 1
        else: dir_y = 1 if abs_pos_y > 0 else 2

    max_steps_possible = int((dt * 1000.0) / STEP_DELAY_MS)

    actual_move_x = min(steps_x, max_steps_possible)
    pos_dir_x = 1 if not INVERT_X else 2
    if actual_move_x > 0:
        if dir_x == pos_dir_x: abs_pos_x += actual_move_x
        else: abs_pos_x -= actual_move_x

    actual_move_y = min(steps_y, max_steps_possible)
    pos_dir_y = 1 if not INVERT_Y else 2
    if actual_move_y > 0:
        if dir_y == pos_dir_y: abs_pos_y += actual_move_y
        else: abs_pos_y -= actual_move_y

    if esp32:
        esp32.write(f"{dir_x},{steps_x},{dir_y},{steps_y}\n".encode())

    display = cv2.resize(frame_raw, (1280, 960), interpolation=cv2.INTER_NEAREST)

    scale_ui = 2.0 
    for name, coords, conf in boxes_to_draw:
        x1, y1, x2, y2 = [int(c * scale_ui) for c in coords]
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(display, f"{name} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    if found:
        tx_f = int(tx * scale_ui)
        ty_f = int(ty * scale_ui)
        cv2.circle(display, (tx_f, ty_f), 12, (0, 0, 255), -1)

    elapsed = time.time() - loop_start
    instant_fps = 1.0 / elapsed if elapsed > 0 else 60
    current_fps = (current_fps * fps_smoothing) + (instant_fps * (1.0 - fps_smoothing))

    cv2.putText(display, f"FPS: {current_fps:.1f} (Local YOLO Mode)", (10, 30), 0, 0.7, (0, 255, 0), 2)
    cv2.putText(display, f"Abs Pos X: {abs_pos_x} Y: {abs_pos_y}", (10, 60), 0, 0.6, (255, 255, 0), 2)

    conn_text = "ESP32: CONNECTED" if esp32 else "ESP32: NOT CONNECTED"
    cv2.putText(display, conn_text, (10, 90), 0, 0.6, (0, 255, 0) if esp32 else (0,0,255), 2)

    if not found:
        cv2.putText(display, "TARGET LOST - HOMING...", (10, 120), 0, 0.7, (0, 165, 255), 2)

    cv2.imshow("Turret Local AI", display)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cam.stop()
cv2.destroyAllWindows()
if esp32:
    esp32.close()
