# 🎯 Dual-Axis AI Turret - Real-Time Object Tracking System

A sophisticated real-time object tracking system that combines computer vision, ESP32 microcontroller control, and dual-axis stepper motors to track and follow any target with precision.

## ✨ Features

### 🧠 Advanced AI Detection
- **Face Detection with Nose Targeting** - Precise aiming at facial features for enhanced accuracy
- **Dynamic Object Detection** - Track any object in real-time using YOLO AI:
  - 👤 People and body parts
  - 🪑 Furniture (chairs, tables, etc.)
  - 🐦 Animals and birds
  - 🎯 Custom objects (define your own targets)
- **Real-Time Processing** - Low-latency detection and motor response
- **Interactive Console Control** - Switch targets on-the-fly without restarting

### 🔧 Hardware Excellence
- **ESP32 Microcontroller** - Powerful embedded processing with dual-axis motor control
- **Dual-Axis Stepper Motors** - Independent Yaw (horizontal) and Pitch (vertical) movement
- **Custom Gear Reductions** - Professional-grade mechanical precision:
  - Yaw axis: 166:24 ratio
  - Pitch axis: 60:20 ratio
- **USB Webcam Support** - Compatible with standard cameras (e.g., Logitech C270)
- **Modular Design** - Easy to adapt and extend

### 💻 Software Stack
- **Python** - Main control script with OpenCV
- **Ultralytics YOLO** - State-of-the-art object detection
- **PySerial** - Seamless ESP32 communication
- **C++ / Arduino** - ESP32 firmware for motor control

---

## 🚀 Quick Start

### Hardware Requirements
- 1x ESP32 Microcontroller
- 2x Stepper Motors (recommended: NEMA 17)
- 2x Stepper Motor Drivers (e.g., DRV8825 or A4988)
- 1x USB Webcam (Logitech C270 or equivalent)
- Power supply (12V recommended for motors)
- Mechanical components for gear reductions and pan-tilt mechanism
- Connecting wires and breadboard

### Software Requirements
```bash
Python 3.7+
OpenCV (cv2)
Ultralytics YOLO (pip install ultralytics)
PySerial (pip install pyserial)
NumPy
