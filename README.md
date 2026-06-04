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

The pipeline classifies 6 fine-grained behaviours into two categories:

| Fine-grained label | Binary label |
|---|---|
| Attentive | High engagement |
| Talking, Head down, Head side, Distracted, On phone | Low engagement |

---

## Attendance — Face Recognition System

### Detection model
**InsightFace `antelopev2`** — SCRFD detector + GLinT-R100 recogniser (ResNet-100, Glint360K, 512-d L2-normalised embeddings)

### How enrollment works

1. Upload a close-up photo/video of the student (or paste a local folder path for multiple clips)
2. Sample **1 frame per second** from the video (up to 30 frames)
3. **Anchor-based tracking** — the first detected face becomes the identity anchor; subsequent frames are only accepted if cosine similarity to anchor ≥ 0.35. This prevents multi-person videos from mixing identities.
4. For each accepted frame, generate **4 embeddings**:
   - Original quality
   - Degraded to **28 px** absolute width (INTER_AREA → Gaussian σ=1 → JPEG q50 → bicubic up)
   - Degraded to **36 px**
   - Degraded to **44 px**
5. All embeddings stored in the gallery + a weighted-mean prototype computed

> The degraded variants bridge the domain gap between close-up enrollment (140–230 px face) and distant classroom faces (14–50 px).

### How attendance marking works

1. Upload a classroom photo
2. Detect all faces using SCRFD at 1280×1280
3. For each detected face, extract a 512-d embedding
4. For each enrolled student, compute:
   `similarity = max(cosine vs prototype, max cosine vs all stored embeddings)`
5. If best similarity ≥ **0.38** → recognized
6. If two faces both match the same student, only the highest-scoring face gets the name — others stay Unknown

### Re-enrollment
Re-enrolling a student with the same name **adds** new embeddings to their existing gallery — does not overwrite. Useful for adding classroom-condition clips on top of selfie enrollment.

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
