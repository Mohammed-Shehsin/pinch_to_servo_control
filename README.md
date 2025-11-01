Pinch-to-Servo Mapping using OpenCV + MediaPipe
=================================================

Description
-----------
This project implements a real-time hand gesture based control system 
where the distance between the thumb tip and index finger tip is used to 
generate a servo angle value (0 to 180 degrees). The concept is simple:
- When fingers touch → angle = 0°
- When fingers separate → angle increases until 180°

This can later be mapped to control actual servo motors via serial communication.

Features
--------
- Real-time hand tracking using MediaPipe Hands
- Pinch detection (thumb tip to index tip)
- Distance normalization using hand width (index MCP to pinky MCP)
- Dynamic linear mapping to angle range [0°, 180°]
- Live UI with angle bar and numeric display
- Manual calibration (set custom MIN/MAX pinch distances)

Controls
--------
- Press 'Z' to set current pinch as ZERO point
- Press 'X' to set current pinch as MAX point
- Press 'Q' to quit

Usage
-----
1. Install dependencies:
    pip install opencv-python mediapipe numpy

2. Run:
    python pinch_to_servo.py

3. Show your right hand facing camera
4. Pinch thumb + index: see angle 0°
5. Spread them: see angle increasing to 180°

Future Extensions
-----------------
- Send angle via serial to Arduino to control servo
- Control 2 servos using 2 different gestures
- Combine with a GUI dashboard
- Use for gesture-based robotic manipulation



Author
------
Created by Shehsin (2025)
