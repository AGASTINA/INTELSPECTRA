# 🛡️ IntelSpectra – AI-Powered Video Intelligence & Threat Detection Platform

> Next-generation AI surveillance platform for real-time threat detection, forensic video analysis, and intelligent security monitoring.

![License](https://img.shields.io/badge/Project-Smart%20India%20Hackathon%202025-blue)
![Python](https://img.shields.io/badge/Python-3.10+-yellow)
![YOLO](https://img.shields.io/badge/YOLO-v11-red)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![Status](https://img.shields.io/badge/Status-Prototype-success)

---

## 📖 Overview

IntelSpectra is an AI-powered surveillance intelligence platform developed for **Smart India Hackathon 2025** under the National Security Guard (NSG) problem statement.

The platform transforms conventional CCTV systems into intelligent surveillance networks capable of detecting weapons, suspicious activities, fire, vehicles, unauthorized persons, and other security threats in real time.

Unlike traditional monitoring systems that rely heavily on human operators, IntelSpectra combines multiple AI models, real-time analytics, forensic search, facial recognition, and predictive intelligence into one unified platform for faster and smarter decision making.

---

## 🎯 Problem Statement

Traditional surveillance systems suffer from several limitations:

- Passive monitoring requiring constant human attention
- Difficulties managing hundreds of camera feeds
- Lack of intelligent threat detection
- Slow forensic investigation
- Poor quality legacy camera footage
- No unified monitoring dashboard

IntelSpectra solves these challenges using AI-powered computer vision and real-time analytics.

---

# ✨ Key Features

### 🎥 Multi-Camera Unified Monitoring

- Live monitoring of unlimited camera feeds
- Unified dashboard
- Multiple layouts
- Centralized surveillance

Supported protocols:

- RTSP
- RTMP
- HTTP / HTTPS
- HLS (M3U8)
- MJPEG
- USB Cameras
- IP Cameras
- Video Files

---

### 🤖 Multi-Modal AI Detection

Runs multiple AI models simultaneously for:

- 🔫 Weapon Detection
- 🔥 Fire Detection
- 🚗 Vehicle Detection
- 👤 Face Recognition
- 🎭 Face Hiding Detection
- 🚶 Suspicious Behaviour Detection
- 🛡️ NSG vs Non-NSG Identification
- 📦 Custom Object Detection

Each model includes:

- Adjustable confidence threshold
- Enable/Disable switch
- Independent processing pipeline

---

### 👁 Face Recognition & Watchlist Tracking

- Face enrollment
- Multiple image registration
- Live face recognition
- Continuous suspect tracking
- Automatic alert generation

Perfect for:

- Wanted criminals
- Missing persons
- VIP security
- Restricted area monitoring

---

### 🗺 Intelligent 2D Building Mapping

Visualizes

- Cameras
- Doors
- Walls
- Security personnel
- Suspect movement

Real-time tracking allows operators to instantly locate detected threats inside buildings.

---

### 📢 Real-Time Threat Alerts

Instant alerts for

- Weapon detection
- Fire detection
- Suspicious movement
- Unauthorized access
- Face match
- Vehicle tracking

Alerts include

- Timestamp
- Camera ID
- Threat confidence
- Snapshot
- Location

---

### 📄 Automated Incident Reports

IntelSpectra automatically generates detailed reports containing

- Event Information
- Detection Summary
- Threat Statistics
- Snapshots
- Detection Timeline
- Camera Details
- Confidence Scores

Ideal for

- Investigation
- Legal documentation
- Operational review
- Intelligence analysis

---

### 📹 Legacy Camera Enhancement

Supports AI-powered enhancement for old CCTV systems using

- Video denoising
- Resolution enhancement
- Sharpening filters
- Noise reduction

No need to replace existing surveillance infrastructure.

---

## ⚙️ System Architecture

```text
                  Camera Sources
        (RTSP • CCTV • Drone • BodyCam)

                      │
                      ▼

            Multi-Protocol Video Gateway

                      │
                      ▼

         AI Processing Engine (YOLOv11)

     ┌────────────┬─────────────┬─────────────┐
     │ Weapon AI  │ Fire AI     │ Vehicle AI │
     ├────────────┼─────────────┼─────────────┤
     │ Face AI    │ Behaviour   │ Tracking AI│
     └────────────┴─────────────┴─────────────┘

                      │
                      ▼

          Threat Fusion & Risk Scoring

                      │
                      ▼

       Alerts • Dashboard • Reports • Archive
```

---

# 🛠 Technology Stack

## AI

- YOLOv11
- PyTorch
- OpenCV
- Computer Vision

## Backend

- Python
- FastAPI

## Streaming

- RTSP
- RTMP
- HTTP
- HTTPS
- HLS
- MJPEG

## Messaging

- Kafka
- Redis

## Deployment

- Intel NPU
- Edge Computing
- Cloud Deployment

## Security

- Authentication
- Secure Access Control
- Role-Based Permissions

---

# 🚀 Core Modules

- Authentication System
- Multi-Camera Manager
- Video Streaming Engine
- AI Detection Engine
- Face Recognition Module
- Watchlist Manager
- Event Management
- Alert Engine
- Incident Reporting
- Building Map Visualization
- Detection Logs
- Archive Scanner

---

# 📊 Applications

🏛 National Security Guard

👮 Police Surveillance

✈ Airport Security

🚆 Railway Protection

🏙 Smart Cities

🏢 Government Buildings

🎭 Public Events

🏫 Educational Institutions

🏭 Industrial Security

---

# 📈 Impact

- Converts passive CCTV into intelligent surveillance
- Reduces operator workload
- Faster emergency response
- Improves public safety
- Supports predictive policing
- Enhances forensic investigations
- Reuses existing surveillance infrastructure
- Enables nationwide scalable deployment

---

# 🔒 Security Features

- Authentication-based access
- Encrypted communication
- Audit logging
- Secure cloud deployment
- Edge AI support
- Privacy-aware architecture

---

# 📂 Project Structure

```
IntelSpectra/

│── frontend/
│── backend/
│── ai_models/
│── face_recognition/
│── streaming/
│── reports/
│── alerts/
│── datasets/
│── weights/
│── utils/
│── docs/
│── README.md
```

---

# 📌 Future Enhancements

- Drone surveillance integration
- Thermal camera support
- Audio anomaly detection
- Large Language Model assisted forensic search
- Cross-camera re-identification
- Predictive threat analytics
- Automatic incident summarization
- Mobile command center
- Satellite surveillance integration

---

# 🏆 Smart India Hackathon 2025

**Problem Statement ID:** SIH25197

**Theme:** AI & ML Enabled Video Analysis and Interpretation

**Category:** Software

Developed as a proposed AI surveillance solution for the National Security Guard (NSG).

---

# 👨‍💻 Team IntelSpectra

Developed for **Smart India Hackathon 2025**

Team: **IntelSpectra**

---

# 📜 License

This project is developed for research, educational, and Smart India Hackathon purposes.

---

⭐ If you found this project interesting, consider giving it a star!
