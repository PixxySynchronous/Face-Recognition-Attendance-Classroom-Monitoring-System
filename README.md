# PRISM AI — Classroom Monitoring System

An AI-powered classroom engagement and attendance monitoring system. It processes classroom video to detect student faces, classify engagement levels, and mark attendance — all through a browser-based dashboard.

---

## What It Does

| Module | What it monitors | How |
|---|---|---|
| **Activity** | Engagement per student every 30 seconds | 3D CNN (ResNet-18) on face crop clips |
| **Engagement** | Body posture, head state, motion, phone use | YOLOv8-pose body keypoints |
| **Cognitive** | Eye gaze, blink rate, facial emotion | dlib 68-pt landmarks + DeepFace |
| **Combined** | All of the above merged | Engagement + Cognitive signals |
| **Classroom** | Action classification with student re-ID | Face re-identification + signal fusion |
| **Attendance** | Who is present in a classroom photo | Face enrollment + cosine similarity matching |

---

## Project Structure

```
deployment/
├── activity_web/             # Flask web application
│   └── backend/
│       ├── app.py            # API routes
│       ├── attendance_service.py  # Face enrollment & matching
│       ├── *_loader.py       # Pipeline loaders
│       ├── static/           # Frontend JS + CSS
│       └── templates/        # HTML
├── ACTIVITY CLASSIFICATION PIPELINE/
│   └── student_activity_pipeline.py   # 3D CNN pipeline
├── ENGAGEMENT PIPELINE/
├── COGNITIVE PIPELINE/
├── COMBINED PIPELINE/
├── CLASSROOM PIPELINE/
├── utils/
│   └── retinaface_detector.py         # SAHI face detection wrapper
├── Activity monitoring/
│   ├── models/best_model/
│   │   └── 3dcnn_r3d18_weighted.pt    # Trained activity classifier (127 MB)
│   └── Training Pipelines/assets/
│       ├── yolo11m.pt                 # Phone detector (39 MB)
│       └── pose_landmarker_lite.task
├── models/retinaface_finetune/        # Fine-tuned RetinaFace ONNX
├── yolov8s-pose.pt                    # Pose estimation
├── yolov8n-pose.pt
├── requirements.txt
├── Procfile                           # Railway / Gunicorn entry point
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

> The first run will also auto-download InsightFace `antelopev2` models (~350 MB) into `~/.insightface/`.

### 4. Set environment variables (optional)

Copy `.env.example` to `.env` and edit as needed:

```bash
cp .env.example .env
```

Key variables:

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8080` | Server port |
| `ACTIVITY_WEB_RUNTIME_DIR` | `activity_web/runtime` | Where uploads and outputs are stored |
| `FLASK_ENV` | `production` | Set to `development` for debug mode |

---

## Running the Website Locally

```bash
# From the deployment/ folder with the virtual environment active:
PORT=8080 gunicorn --bind 0.0.0.0:8080 --timeout 300 --workers 1 activity_web.backend.app:app
```

Then open **http://localhost:8080** in your browser.

For development with auto-reload:
```bash
FLASK_ENV=development FLASK_APP=activity_web.backend.app flask run --port 8080
```

---

## Using the Website

### Enroll a student (Attendance tab)

1. Go to the **Attendance** section
2. Enter the student's name
3. **Upload files** — upload one or more close-up photos/videos of the student's face
   - OR —
   **From folder path** — paste the local path to a folder containing video clips of the student (e.g. `/Users/you/jai_clips`)
4. Click **Enroll student**

> Re-enrolling the same name adds embeddings to the existing gallery — it does not overwrite.

### Mark attendance

1. Upload a classroom photo
2. Click **Mark attendance**
3. The system detects all faces, matches against the enrolled roster, and returns a marked image with names and confidence scores

### Run activity monitoring

1. Upload a classroom video (MP4/MOV)
2. Select a pipeline: **Activity**, **Engagement**, **Cognitive**, **Combined**, or **Classroom**
3. Click **Analyse**
4. Results include per-student engagement labels, confidence scores, and downloadable face-crop clips

---

## Deploying to Railway

### One-click via Procfile

Railway picks up `Procfile` automatically:

```
web: gunicorn --bind 0.0.0.0:$PORT activity_web.backend.app:app
```

### Steps

1. Push this folder to a GitHub repository (with Git LFS for model files)
2. Create a new Railway project → **Deploy from GitHub repo**
3. Set environment variables in Railway dashboard (copy from `.env.example`)
4. Railway builds and deploys automatically

### Persistent storage

Railway's ephemeral filesystem means uploaded videos and outputs are lost on redeploy. Set these variables to point to a Railway Volume:

```
ACTIVITY_WEB_RUNTIME_DIR=/data/runtime
ACTIVITY_WEB_UPLOAD_DIR=/data/runtime/uploads
ACTIVITY_WEB_OUTPUT_DIR=/data/runtime/outputs
ACTIVITY_WEB_ATTENDANCE_DIR=/data/runtime/attendance
```

---

## Face Recognition Details

**Model:** InsightFace `antelopev2` — ResNet-100 backbone, Glint360K trained, 512-d L2-normalised embeddings

**Enrollment pipeline:**
1. Sample 1 frame/second from the enrollment video (up to 30 frames)
2. Lock onto the first detected face as the identity anchor
3. Accept subsequent frames if cosine similarity to anchor ≥ 0.35
4. For each accepted frame, also generate degraded variants at 28 / 36 / 44 px absolute width (INTER_AREA → Gaussian σ=1 → JPEG q50 → bicubic up) to bridge the gap between close-up enrollment and distant classroom faces
5. Store all embeddings + a weighted-mean prototype

**Matching:**
- Threshold: **0.38** cosine similarity
- Match = max(cosine vs prototype, max cosine vs all stored embeddings)
- If two faces match the same student, only the highest-scoring face gets the label

---

## Activity Classification Model

| Model | Accuracy | High engagement F1 | Macro F1 |
|---|---|---|---|
| 3D CNN R3D-18 (class-weighted) | **93.2%** | **79.3%** | **87.6%** |

Evaluated on 176 labeled student clips (binary: high / low engagement).  
Training dataset: individual student face-crop clips labeled across 6 fine-grained categories (attentive, talking, head_down, head_side, distracted, phone).

---

## Requirements

See `requirements.txt`. Key dependencies:

```
Flask
gunicorn
numpy
opencv-python-headless
ultralytics          # YOLOv8 / YOLO11
insightface          # face detection + recognition
torch
torchvision          # 3D CNN (r3d_18)
dlib                 # cognitive pipeline landmarks
deepface             # emotion recognition
```

> `dlib` requires CMake and a C++ compiler. On macOS: `brew install cmake`. On Ubuntu: `apt install cmake build-essential`.
