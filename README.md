# PRISM AI — Classroom Monitoring System

An AI-powered classroom engagement and attendance monitoring system with a browser-based dashboard.

---

## What It Does

| Tab | Function |
|---|---|
| **Classroom Monitoring** | Upload a classroom video → detect and track every student → classify engagement every 30 seconds using a fine-tuned 3D CNN → per-student timeline with clips |
| **Attendance** | Enroll students from photos/videos → mark attendance from a classroom photo using face recognition |

---

## Activity Classification Model

Evaluated on 176 labeled student clips — binary: **high engagement** vs **low engagement**.

| Model | Accuracy | High Engage F1 | Macro F1 |
|---|---|---|---|
| **3D CNN R3D-18 (class-weighted)** | **93.2%** | **79.3%** | **87.6%** |

---

## Running Locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# macOS — always include OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES PORT=8080 \
  .venv/bin/gunicorn --bind 0.0.0.0:8080 --timeout 600 --workers 1 \
  activity_web.backend.app:app
```

Open **http://localhost:8080**

---

## Deployment

See [`deployment/README.md`](deployment/README.md) for full Railway deployment instructions.

---

## Repository Structure

```
ACTIVITY CLASSIFICATION PIPELINE/   3D CNN engagement classifier
CLASSROOM PIPELINE/                  Main classroom analysis pipeline
ENGAGEMENT PIPELINE/                 YOLOv8-pose engagement signals
COGNITIVE PIPELINE/                  EAR / gaze / emotion (dlib + DeepFace)
COMBINED PIPELINE/                   Merged engagement + cognitive
activity_web/                        Flask web app (2 tabs: Classroom + Attendance)
utils/                               SAHI face detection wrapper
Activity monitoring/models/          Trained model weights
deployment/                          Self-contained Railway deployment build
```
