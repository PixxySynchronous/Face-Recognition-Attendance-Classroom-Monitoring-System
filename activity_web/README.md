# Activity Classification Web App

This adds a basic Flask backend and a browser frontend on top of the existing classroom activity classification pipeline.

## What it does

- Upload a classroom video from the browser
- Run the existing `StudentActivityPipeline`
- Return the generated JSON summary, CSV, and annotated video download links
- Show per-clip predictions in a table
- Enroll students from photos or videos using local embeddings derived from fine-tuned RetinaFace-aligned face crops
- Mark attendance from a classroom photo with fine-tuned RetinaFace detection and local embedding matching
- Delete an enrolled student from the roster and remove their stored attendance rows

## Tabs

- `Activity Monitoring` runs the activity classification pipeline.
- `Attendance` lets you enroll students and mark attendance from a classroom photo.

## Run it

From the repo root:

```bash
source /Users/satyam/Desktop/classroom-ai-project/.venv/bin/activate
python -m activity_web.backend.app
```

Then open `http://127.0.0.1:5000`.

### Important: ffmpeg for browser playback

The pipeline saves per-student clips as MP4 files. To ensure those clips are encoded in H.264
so modern browsers can play them inline, install `ffmpeg` and make it available on your PATH
before running the server. If `ffmpeg` is not found the app will fall back to a basic encoder
which may produce MP4 files that some browsers cannot decode.

Installation examples:

- Windows (Chocolatey): `choco install ffmpeg`
- macOS (Homebrew): `brew install ffmpeg`
- Ubuntu/Debian: `sudo apt install ffmpeg`


## Notes

- The app reuses `ACTIVITY CLASSIFICATION PIPELINE/student_activity_pipeline.py` directly.
- Output files are written to `activity_web/runtime/`.
- Large videos can take a while because the pipeline performs full face detection and classification.
