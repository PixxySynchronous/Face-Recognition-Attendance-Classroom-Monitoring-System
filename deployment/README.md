# PRISM AI — Classroom Monitoring System

An AI-powered classroom engagement and attendance monitoring system. It processes classroom video to detect, track, and classify student engagement — and marks attendance from a single classroom photo — all through a browser-based dashboard.

---

## What It Does

### Classroom Monitoring tab
Upload a classroom video and the system:
- Detects every student's face every few seconds using SCRFD (InsightFace)
- Re-identifies the same student across windows using face embeddings + **seat-position tracking** (students stay in their seats)
- Classifies each student's action per window: Attentive / Writing / Talking / On Phone / Sleeping / Distracted
- Outputs a per-student timeline, action breakdown bar, and per-window face-crop clips
- Downloadable summary JSON + CSV

### Attendance tab
- **Enroll students** from a close-up selfie video or photo (upload files or paste a local folder path)
- **Mark attendance** from a classroom photo — detects all faces, matches against enrolled roster, returns a marked image with names and confidence scores

---

## Project Structure

```
deployment/
├── activity_web/                  # Flask web application
│   └── backend/
│       ├── app.py                 # API routes
│       ├── attendance_service.py  # Face enrollment & matching
│       ├── *_loader.py            # Pipeline loaders
│       ├── static/                # Frontend JS + CSS
│       └── templates/             # HTML (2 tabs: Classroom + Attendance)
├── ACTIVITY CLASSIFICATION PIPELINE/
│   └── student_activity_pipeline.py   # 3D CNN engagement classifier
├── ENGAGEMENT PIPELINE/
├── COGNITIVE PIPELINE/
├── COMBINED PIPELINE/
├── CLASSROOM PIPELINE/
│   └── classroom_pipeline.py          # Main classroom analysis pipeline
├── utils/
│   └── retinaface_detector.py         # SAHI face detection wrapper
├── Activity monitoring/
│   ├── models/best_model/
│   │   └── 3dcnn_r3d18_weighted.pt    # Trained activity classifier (127 MB)
│   └── Training Pipelines/assets/
│       ├── yolo11m.pt                 # Phone detector (39 MB)
│       └── pose_landmarker_lite.task
├── models/retinaface_finetune/        # Fine-tuned RetinaFace ONNX
├── yolov8s-pose.pt
├── yolov8n-pose.pt
├── requirements.txt
├── Procfile
└── Dockerfile.railway
```

---

## Prerequisites

- **Python 3.10–3.12** (recommended; Python 3.14 has some package compatibility issues)
- **Git LFS** — required to clone the model files
- **ffmpeg** — for video clip encoding (optional but recommended)

---

## Local Setup

### 1. Clone the repository

```bash
git lfs install          # ensure LFS is active before cloning
git clone <repo-url>
cd deployment
```

> If you cloned before running `git lfs install`, run `git lfs pull` to download the model files.

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> First run will auto-download InsightFace `antelopev2` models (~350 MB) into `~/.insightface/`.

### 4. Set environment variables (optional)

```bash
cp .env.example .env
# edit .env as needed
```

---

## Running the Website Locally

```bash
# Standard
PORT=8080 gunicorn --bind 0.0.0.0:8080 --timeout 600 --workers 1 activity_web.backend.app:app

# macOS (required — prevents fork-safety crash with PyTorch)
OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES PORT=8080 gunicorn \
  --bind 0.0.0.0:8080 --timeout 600 --workers 1 \
  activity_web.backend.app:app
```

Then open **http://localhost:8080**

> **macOS note:** Always include `OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES` on macOS, otherwise the worker will crash mid-processing when PyTorch and Objective-C libraries conflict during fork.

---

## Using the Website

### Classroom Monitoring

1. Go to the **Classroom Monitoring** tab
2. Upload a classroom video (MP4 / MOV / AVI)
3. Click **Analyse classroom**
4. Results show:
   - Class-wide action distribution bar
   - Per-student collapsible cards with timeline, action labels, and face-crop clips
   - Download links for summary JSON and CSV

### Enroll a student (Attendance tab)

1. Go to the **Attendance** tab → **Enroll student**
2. Enter the student's name
3. Choose an upload method:
   - **Upload files** — select one or more close-up photos/videos
   - **From folder path** — paste a local folder path (e.g. `/Users/you/jai_clips`) — all video/image files in that folder are used automatically
4. Click **Enroll student**

> Re-enrolling the same name **adds** embeddings to the existing gallery — it does not overwrite.

### Mark attendance

1. Upload a classroom photo under **Mark attendance**
2. Click **Mark attendance**
3. Returns a marked image with recognised student names and confidence scores, plus an attendance log

---

## Deploying to Railway

### Steps

1. Push this folder to a GitHub repository (with Git LFS for `*.pt` and `*.onnx.data` files)
2. Create a new Railway project → **Deploy from GitHub repo**
3. Set environment variables in Railway dashboard (copy from `.env.example`)
4. Railway picks up `Procfile` automatically:

```
web: gunicorn --bind 0.0.0.0:$PORT activity_web.backend.app:app
```

### Persistent storage

Railway's ephemeral filesystem loses uploaded files on redeploy. Set these variables to point to a Railway Volume:

```
ACTIVITY_WEB_RUNTIME_DIR=/data/runtime
ACTIVITY_WEB_UPLOAD_DIR=/data/runtime/uploads
ACTIVITY_WEB_OUTPUT_DIR=/data/runtime/outputs
ACTIVITY_WEB_ATTENDANCE_DIR=/data/runtime/attendance
```

---

## Face Recognition Details

**Model:** InsightFace `antelopev2` — ResNet-100, Glint360K, 512-d L2-normalised embeddings

**Enrollment:**
1. Sample 1 frame/second from the enrollment video (up to 30 frames)
2. Lock on first detected face as anchor; accept subsequent frames if cosine similarity ≥ 0.35
3. For each accepted frame, also store degraded variants at **28 / 36 / 44 px** absolute width (INTER_AREA → Gaussian σ=1 → JPEG q50 → bicubic up) — bridges the gap between close-up enrollment and distant classroom faces
4. Store all embeddings + weighted-mean prototype

**Matching at attendance time:**
- Threshold: **0.38** cosine similarity
- Compares against prototype AND all stored individual embeddings (takes max)
- Only the highest-scoring face per student gets the name label

---

## Activity Classification Model

| Model | Accuracy | High engagement F1 | Macro F1 |
|---|---|---|---|
| 3D CNN R3D-18 (class-weighted) | **93.2%** | **79.3%** | **87.6%** |

Evaluated on 176 labeled student clips — binary: **high engagement** (attentive) vs **low engagement** (talking, head down, distracted, on phone, sleeping).

---

## Student Re-Identification (Classroom Pipeline)

Students are tracked across burst windows using a **two-stage matching strategy**:

1. **Position-first** — find the bank entry whose last known seat position is closest to the current track. If distance < 12% of frame width AND cosine similarity ≥ 0.28 → assign that student. Students almost never change seats during a lecture.
2. **Embedding fallback** — if no positional match, use pure cosine similarity ≥ 0.40.

This prevents the common mis-assignment where two students with similar faces swap IDs between windows.

---

## Requirements

Key dependencies (see `requirements.txt`):

```
Flask / gunicorn
numpy / opencv-python-headless
ultralytics        # YOLOv8 / YOLO11
insightface        # face detection + recognition
torch / torchvision
dlib               # cognitive pipeline landmarks
deepface           # emotion recognition
```

> `dlib` requires CMake. macOS: `brew install cmake`. Ubuntu: `apt install cmake build-essential`.
