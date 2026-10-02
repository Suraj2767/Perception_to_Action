# Perception-to-Action Robot System

## Overview

This project implements a perception-to-action robotics pipeline in a PyBullet simulation.

A KUKA robotic arm detects a randomly placed red object using a simulated camera and OpenCV-based color detection. The detected image coordinates are converted into 3D world coordinates, and inverse kinematics (IK) is used to move the robotic arm toward the object.

The system demonstrates a closed-loop perception-to-action workflow where the object position changes between runs and the robot automatically reacts to the detected position.

## Pipeline

Simulated Camera
        ↓
RGB Image
        ↓
OpenCV Red Color Detection
        ↓
Object Centroid
        ↓
Pixel → 3D World Coordinate
        ↓
Inverse Kinematics
        ↓
KUKA Robotic Arm
        ↓
Touch / Collect Object

## Features

- PyBullet physics simulation
- KUKA iiwa robotic arm
- Random object placement on every run
- Simulated RGB camera
- OpenCV HSV color thresholding
- Contour-based object detection
- Automatic pixel-to-world coordinate conversion
- Inverse kinematics for robot motion
- Closed-loop perception-to-action behavior
- Multiple consecutive runs with different object positions

## Technologies Used

- Python
- PyBullet
- OpenCV
- NumPy
- Inverse Kinematics
- Computer Vision
- Robotics Simulation

## Robot

The project uses the KUKA iiwa robotic arm available in the PyBullet data assets.

The robot is controlled using PyBullet inverse kinematics to move its end effector toward the detected object.

## Object Detection

The simulated camera captures an RGB image of the workspace.

OpenCV is used to detect the red object using HSV color thresholding.

The detected contour provides the object's image centroid:

```text
(cx, cy)

Project Structure
Perception_to_Action/
│
├── perception_to_action.py
├── requirements.txt
└── README.md