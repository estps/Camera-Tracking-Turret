AI Tracking Software (Local & Networked)

This document covers the Python-based computer vision and tracking logic for the Dual-Axis AI Turret.

Instead of forcing the microcontroller to handle complex math and AI models, this software runs on a PC or laptop. It processes the live video feed, identifies targets, and calculates the exact physical movements required, sending simple step commands down to the ESP32.

The project supports two distinct modes of operation depending on your hardware:

Local Tracking Mode: Runs the YOLO11 AI models directly on your local machine (seamlessly auto-detecting CPU or iGPU).

External/Networked Tracking Mode: Offloads the heavy AI processing to a dedicated GPU server (e.g., a machine with an RTX 4060) via a high-speed TCP socket, keeping the local laptop lightweight.

Core Features

Dual Architecture: Choose between standalone local execution or a decoupled remote-GPU client/server architecture.

Asynchronous Processing: Separates the camera feed, AI detection, and networking into dedicated threads to maintain a smooth 60 FPS video stream.

Exact Positional Math: Translates pixel errors into physical degrees using the camera's specific Field of View, then applies custom gear ratios (166:24 and 60:20) to calculate precise motor microsteps.

Absolute Homing: Tracks the turret's position relative to its power-on origin and automatically generates a return path to center itself when a target is lost.

Dynamic Targeting: Uses Ultralytics YOLO to track custom objects or human poses (noses). Targets can be switched on the fly via the console.

Audio Feedback: Utilizes text-to-speech to announce target locks and confidence scores without flooding the terminal.

Software Requirements

You will need Python 3 installed along with the following libraries:

opencv-python (cv2)

ultralytics

pyserial

numpy

pyttsx3

Setup and Usage

Hardware Connection: Ensure your ESP32 is plugged in and accessible (default is /dev/ttyACM0 on Linux).

Camera: Connect your USB webcam (default is a Logitech C270). If your system assigns a different index, update the CameraStream initialization in the code.

Execution: Run the desired Python script based on your setup:

For local processing: Run the local tracking script directly.

For networked processing: Start the AI server script on your GPU machine, then run the bridge script on the laptop connected to the turret.

Targeting: By default, the turret tracks "person". You can type a new target name directly into the running console to switch targets seamlessly.

Calibration

If you change your camera or gear setup, update the hardware calibration section at the top of the Python script. You must define the new Field of View (FOV_X, FOV_Y) and gear ratios for the math to remain accurate.
