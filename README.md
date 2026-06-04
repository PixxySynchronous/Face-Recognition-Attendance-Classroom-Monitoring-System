# PRISM AI — Classroom Monitoring System

An AI-powered classroom engagement and attendance monitoring system with a browser-based dashboard.

---

## What It Does

| Tab | Function |
|---|---|
| **Classroom Monitoring** | Upload a classroom video → detect and track every student → classify engagement using EAR, MAR, YOLO-pose, gaze and emotion signals → per-student timeline with clips |
| **Attendance** | Enroll students from photos/videos → mark attendance from a classroom photo using face recognition |

---

## Classroom Monitoring — Pipeline

The **Classroom pipeline** (`CLASSROOM PIPELINE/classroom_pipeline.py`) uses the following signals per student per window:

| Signal | How |
|---|---|
| **EAR** (Eye Aspect Ratio) | dlib 68-pt landmarks → eyes open/closed, blink detection |
| **MAR** (Mouth Aspect Ratio) | dlib 68-pt landmarks → talking detection |
| **Head pose / gaze** | solvePnP on dlib landmarks → looking centre / left / right / up / down |
| **Body keypoints** | YOLOv8-pose → head state, posture, motion |
| **Phone detection** | YOLO → on phone → immediately low engagement |
| **Emotion** | DeepFace → happy / neutral / surprise / sad / angry |
| **Face re-ID** | InsightFace embeddings + seat-position tracking across windows |

Actions classified per student: **Attentive / Writing / Talking / On Phone / Sleeping / Distracted**

---

## Attendance — Face Recognition System

### Detection & recognition model
**InsightFace `antelopev2`** — SCRFD-10G face detector + GLinT-R100 recogniser (ResNet-100, Glint360K, 512-d L2-normalised embeddings), running at det_size=1280×1280

> `utils/retinaface_detector.py` also contains a **SAHI + MTCNN + buffalo_l** wrapper used by the classroom and cognitive pipelines for better small-face recall.

### How enrollment works

1. Upload a close-up photo/video (or paste a local folder path for multiple clips)
2. Sample **1 frame per second** (up to 30 frames)
3. **Anchor-based tracking** — first detected face is the identity anchor; subsequent frames accepted only if cosine similarity ≥ 0.35 (prevents multi-person videos from mixing identities)
4. For each accepted frame, generate **4 embeddings**:
   - Original quality
   - Degraded to **28 px** (INTER_AREA → Gaussian σ=1 → JPEG q50 → bicubic up)
   - Degraded to **36 px**
   - Degraded to **44 px**
5. All embeddings stored + weighted-mean prototype computed

> Degraded variants bridge the domain gap between close-up enrollment (140–230 px face) and distant classroom faces (14–50 px).

### How attendance marking works

1. Upload a classroom photo
2. Detect all faces (SCRFD 1280×1280)
3. For each face: `similarity = max(cosine vs prototype, max cosine vs all stored embeddings)`
4. If best similarity ≥ **0.38** → recognized
5. Only the highest-scoring face per student gets the name label

### Re-enrollment
Re-enrolling the same name **adds** embeddings to the existing gallery — does not overwrite.

---

## Running Locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Download YOLO model weights (Git LFS pointers — run once after cloning)
python download_models.py

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
CLASSROOM PIPELINE/                  Main classroom analysis pipeline (EAR/MAR/YOLO/emotion)
ENGAGEMENT PIPELINE/                 YOLOv8-pose engagement signals
COGNITIVE PIPELINE/                  EAR / gaze / emotion (dlib + DeepFace)
COMBINED PIPELINE/                   Merged engagement + cognitive
activity_web/                        Flask web app (2 tabs: Classroom + Attendance)
utils/                               SAHI + MTCNN + buffalo_l face detection wrapper
Activity monitoring/models/          Trained model weights
deployment/                          Self-contained Railway deployment build
```
