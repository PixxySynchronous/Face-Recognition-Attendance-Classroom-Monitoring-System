# Session summary — AdaFace migration + Classroom Monitoring overhaul

Everything below happened in one continuous session on `prism-ai-main`. Written so a
fresh session (human or AI) can pick this up without re-deriving context.

## Status at end of session

- **Pushed to `origin` (GitHub `PRISM-AI-REPO`) and `hf` (Hugging Face Space
  `pixxysynchronous/Prismai`)**: only the Attendance AdaFace backbone swap
  (commits `cd43f50`, `c19f17a`, `8ed7102`). This is live.
- **Local only, not pushed**: everything else in this document — UI updates, the new
  demo image, the enrollment-sampling fix, and the entire Classroom Monitoring
  pipeline overhaul. Run `git status` / `git diff --stat` to see the exact file list.
- Three point-in-time backups of `CLASSROOM PIPELINE/classroom_pipeline.py` exist
  next to the real file, in case any of the changes below need reverting:
  - `classroom_pipeline.py.bak` — original, pre-session (dlib present in code but never
    actually installed/working, glintr100 backbone, sequential identity-resolution bug).
  - `classroom_pipeline.py.bak2_before_landmarks` — after the merge-bug fix + AdaFace
    swap + roster integration, before the dlib→InsightFace landmark swap.
  - `classroom_pipeline.py.bak3_before_mouth_signal` — after the landmark swap +
    roster threshold/margin tuning + merge-by-name, before switching attentiveness
    detection from EAR to mouth-closed percentage.

---

## 1. Attendance: recognition backbone swapped glintr100 → AdaFace IR-101

**Why**: eval work in `prism-ai-main-copy` (a separate scratch/eval repo) showed
AdaFace IR-101 (WebFace12M) gives cleaner genuine/impostor separation than the
previous glintr100/antelopev2 backbone on real classroom photos.

**What changed** (`activity_web/backend/attendance_service.py`,
new `utils/adaface_backbone.py`):
- Detection stays InsightFace (SCRFD/antelopev2); only the recognition/embedding
  step changed. Faces are aligned via `insightface.utils.face_align.norm_crop`
  then embedded with AdaFace.
- `FACE_SIMILARITY_THRESHOLD` moved from glintr100's **0.38** to AdaFace's
  **0.28** — this is AdaFace's own p99 impostor similarity, derived from
  `eval/impostor_scope_eval.py` in the scratch repo against a human-labeled clean
  set (NOT a guess — a quick eval with a noisier default label set gave a
  different p99=0.35, but the more rigorous script's 0.28 was used since it
  came from cleaner labels).
- `ANCHOR_CONSISTENCY_THRESHOLD` (0.35) and `ENROLLMENT_OUTLIER_SIM_THRESHOLD` (0.50)
  were left unchanged from glintr100 — `eval/build_gallery.py` in the scratch repo
  used the same values when building the AdaFace comparison gallery, so there was
  no evidence they needed to move.
- Existing enrolled students auto-re-embed under the new backbone on next service
  startup via the pre-existing `embedding_model` migration hook — no new
  migration code was needed, it already existed for this purpose.
- Model weights (261MB) are **not** committed to git — fetched at build time via
  `huggingface_hub`, same pattern as the existing YOLO weight downloads in
  `download_models.py`. Added `huggingface_hub` to `requirements.txt`.

**Verification**: real classroom photo
(`20260302_102525__f00000300.jpg`) — Aniket 0.65, Kartik 0.57, Gourav, Rishabh all
correctly recognized in green, 7 strangers correctly red/"Unknown" at 0.13–0.22
similarity (well under the 0.28 threshold).

**Deploy hiccup + fix**: pushing to the `hf` remote initially failed because it has
completely separate git history from `origin` (`git commit-tree` was used to push a
clean commit parented on `hf/main`'s own tip, rather than merging origin's full
history in, which would have dragged in old already-deleted binary blobs `hf`'s
server rejects). A second issue: the Docker build broke because the merge resolution
had accidentally stopped tracking `demo_classroom.jpg` on `origin` — the Dockerfile
fetches that file from `origin`'s GitHub raw URL at build time, so removing it from
origin (even though it correctly stays untracked on `hf`, which has its own binary
policy) 404'd the build. Fixed by restoring the file to `origin`'s tracking and
re-pushing a no-op commit to `hf` to retrigger the build.

**Known latent risk (deferred, not fixed)**: `mark_attendance`'s incremental gallery
growth (auto-adds any match ≥0.60 similarity into that student's stored embeddings)
pre-dates this session but is more dangerous under AdaFace's compressed similarity
range. Repeated testing during this session visibly poisoned a test gallery
(a stranger falsely matched at 0.65, then re-matched itself at 1.0 next run). User
explicitly said to leave this as-is for now — flagging again here for the next session.

---

## 2. Attendance UI updates (local only)

All in `activity_web/backend/templates/index.html`,
`activity_web/backend/static/{app.js,styles.css}`:

- **Attendance tab now opens by default** (was Classroom Monitoring).
- **New "How it Works" tab** — two cards, one per pipeline (Attendance, Classroom
  Monitoring), written to describe what the code actually does, including honest
  callouts of known limitations (see §5).
- **Fixed outdated/wrong copy**: hero text and the roster card used to claim
  detection used "the fine-tuned RetinaFace ONNX model" — wrong; it's always been
  InsightFace SCRFD. Updated to correctly describe InsightFace detection + AdaFace
  recognition.
- **New "unknown faces" viewer**: after Mark Attendance or the demo run, a
  "Show unknown faces (N)" button reveals a grid of zoomed-in crops of every
  unrecognized face with its similarity score — cropped client-side via `<canvas>`
  from the already-loaded marked photo, no extra server round-trip. Backend change:
  `mark_attendance` now returns `unknown_faces_detail: [{bbox, similarity}, ...]`
  alongside the existing `unknown_faces` count.
- **Fixed a CSS Grid layout bug**: when "Mark attendance" results got tall (big
  image + long list), CSS Grid stretched the shorter "Enroll student" card to match
  and its content spread out awkwardly. Fixed with `align-items: start` on the grid
  and `align-content: start` on the cards.

---

## 3. Enrollment frame-sampling fix (local only)

**Bug found while testing**: `attendance_service.py`'s enrollment sampling used a
fixed "1 frame per second" interval. A short ~8.6s enrollment video (JAI's) only
yielded 9 candidate frames vs. ~23 for a 22s video (Aniket's) — before any
rejection even happened, short clips were structurally starved.

**Fix**: `_sample_frames_for_enrollment` now derives the step from
`total_frames // MAX_ENROLLMENT_FRAMES` (target 30), spreading up to 30 samples
evenly across the whole clip regardless of duration or fps. Removed the now-unused
`ENROLLMENT_SAMPLE_INTERVAL_S` constant.

**Verified**: JAI's clip went from 9→30 sampled frames (6→15 accepted embeddings);
Aniket's went from 23→30 sampled frames.

---

## 4. New demo image selected (local only)

The old bundled `demo_classroom.jpg` didn't contain any of the 5 enrolled students
except (as later confirmed by the user) Aniket. Built a browsable gallery Artifact
of 65 real candidate photos from the labeled classroom dataset
(`classroom datset/retinaface_1fps/val/images`) where ≥3 of the 5 enrolled students
were recognized, cross-checked against human-verified ground-truth labels
(`eval/student_labels_clean.json` in the scratch repo) to flag false positives.

Note: an initial scan found 189 "qualifying" images, but that number was inflated
by attendance's incremental-gallery-growth bug (§1) compounding across repeated
test runs — a clean, uncontaminated rerun found the real number was **65** (35
fully clean, 30 with at least one flagged false positive).

**Chosen image**: `20260302_104928__f00003000.jpg` — Rishabh (0.51), Gourav (0.36),
Kartik (0.47) all verified correct against ground truth, 14 strangers correctly
left "Unknown," zero false positives. Now at
`activity_web/backend/static/demo_classroom.jpg`.

---

## 5. Classroom Monitoring pipeline overhaul (local only)

All in `CLASSROOM PIPELINE/classroom_pipeline.py` unless noted. This pipeline is
separate from Attendance — it builds anonymous per-video student tracks and an
engagement timeline from an uploaded classroom video.

### 5a. Fixed: different real people merging into one identity

**Root cause**: `_resolve_identity` was called once per track, sequentially, inside
the per-track loop. Nothing stopped two different real people visible in the *same*
24-frame burst from both independently matching (and blending into) the same
existing anonymous identity — the same class of bug already fixed in Attendance's
`mark_attendance` via its `best_per_student` one-per-window dedup.

**Fix**: identity resolution now happens once per burst as a batch step: build all
`(track, identity, similarity)` candidates, sort by similarity descending, and
greedily assign one-to-one (a track or an identity can be claimed at most once per
burst). Unclaimed tracks become new identities.

**Verified**: a synthetic composite video (Aniket's face left half, Kartik's face
right half, same frame throughout, built by hconcat-ing frames from their real
enrollment videos) resolved to **2 distinct students**, not 1, both through direct
pipeline calls and through the real `/api/classroom/process` web endpoint.

### 5b. Recognition backbone swapped to AdaFace (same as Attendance)

- `FaceAnalysis(allowed_modules=["detection"])` — dropped `"recognition"`.
- Faces aligned via `norm_crop` + embedded with the same `utils/adaface_backbone.py`
  wrapper used by Attendance.
- `IDENTITY_THRESHOLD` (within-video re-identification, *not* the roster-match
  threshold) moved from glintr100's 0.40 to **0.35** — a judgment call (no dedicated
  eval exists for this specific threshold), chosen conservatively between the
  Attendance roster threshold and the old glintr100 value, erring toward fewer
  false merges since a merge is worse here than a split.

### 5c. Roster recognition — real names for enrolled students

New file `utils/roster_match.py`: a small, **read-only** lookup against
`attendance_store.json` (no gallery-growth side effects, unlike Attendance's
`match_student`) so the Classroom pipeline doesn't need Flask package imports or
risk mutating the enrollment gallery from an anonymous video-analysis context.

- `ClassroomPipeline.__init__(roster_path=...)` — optional; if omitted, roster
  recognition is skipped entirely (keeps standalone/CLI usage working without a
  Flask context). `activity_web/backend/classroom_loader.py` wires in the real
  path from `config.ATTENDANCE_DIR`.
- `recognized_name` flows through to the summary JSON, CSV, and the frontend
  (`activity_web/backend/static/app.js`'s `renderClassroomStudents` shows the real
  name with the anonymous `student_00N` label as secondary text).
- **Threshold tuning**: found that a real stranger got matched to "Kartik" in
  testing, *and* separately that the real Kartik sometimes got split into two
  different anonymous IDs that each independently matched him via the roster
  (because within-video re-ID, using one evolving prototype, disagreed with
  roster-matching, which compares against Kartik's full ~80-embedding gallery).
  Two independent fixes:
  1. **Merge duplicate names**: in `_aggregate`, anonymous IDs that both
     confidently match the same enrolled student are now merged into one entry
     before computing stats. Unit-tested with synthetic records — two IDs both
     "Kartik" merged into one (windows summed), an unrelated unrecognized ID
     stayed separate.
  2. **Stricter roster threshold + margin**: `DEFAULT_MATCH_THRESHOLD` raised
     0.28→**0.35** and a new `DEFAULT_MATCH_MARGIN=0.05` requires the best match
     to beat the second-best candidate student by at least that much. This only
     affects `utils/roster_match.py` (video recognition) — Attendance's own
     photo-based `match_student`/0.28 threshold is untouched and unaffected.

### 5d. dlib → InsightFace 106-point landmarks (the big one)

**Discovery**: `dlib` was never actually installed in this environment — not in
`requirements.txt`, and `cmake` isn't even available to build it. This meant
`DLIB_AVAILABLE=False` the entire time, and every EAR/MAR/gaze-based signal
(blink/sleep detection, real mouth-movement talking detection, gaze-based
attention) had been **silently disabled** — not mistuned, entirely dead — with only
a console warning at startup that's easy to miss. This is very likely why activity
classification felt "very inconsistent": half the signal pipeline was never running
at all.

**Fix**: switched to InsightFace's own `landmark_2d_106` model — already bundled
with the same detector already in use, so zero new dependencies. The 106-point
scheme uses different point indices than dlib's 68-point scheme, so the mapping
was derived **empirically**: plotted and numbered all 106 points on a real face
(`/tmp` scratch script, not committed) rather than trusting memorized docs. Confirmed
layout: 0–32 jaw contour, 33–42/87–96 the two eyes (structurally offset by exactly
54 — confirmed numerically), 43–51/97–105 the two eyebrows, 52–61 outer mouth,
62–71 inner mouth, 72–86 nose. Derived ordered index subsets
(`_EYE_LEFT_IDX_106`, `_EYE_RIGHT_IDX_106`, `_MOUTH_INNER_IDX_106`,
`_LANDMARK_2D_IDX`) that plug into the *same, unchanged* `_compute_ear` /
`_compute_mar` / `_head_pose` formulas the dlib version used, just fed differently-
indexed points. Removed all dlib code (`_ensure_dlib_model`, the 68-point index
constants, the `bz2`/`urllib.request` imports that existed only to download the
dlib model).

### 5e. Action classification recalibration against real labeled data

The user has a separate labeled dataset:
`~/Desktop/activity_dataset/datasets/annotations_split2*.csv` — 1175 short
(~2.5s, 320×320) single-person track clips labeled with 6 real action categories
(attentive, distracted, talking, phone, head_down, head_side) plus skip/transition
(excluded from evaluation). A prior pass by the user also produced a **binary**
mapping: only `attentive` → `high_engagement`; everything else (including talking
and head_down) → `low_engagement`
(`annotations_split2_high_test_{train,test}.csv`).

**First pass (72-clip stratified sample, 12 per category)** — smoking gun: across
every single sample, the classifier *only ever* predicted "Distracted," "Talking,"
or "NO_DETECTION." "Attentive," "On Phone," "Sleeping," and "Writing" were never
predicted once, regardless of ground truth.

**Root causes, confirmed with raw per-window signal data** (not guessed):

1. **Phone detection is completely dead.** `phone_pct` was 0.0 in *every* sample,
   including all "phone" ground-truth clips. Direct test: ran the actual phone
   detector (`Activity monitoring/Training Pipelines/assets/yolo11m.pt` — a
   generic, un-fine-tuned COCO YOLO11m model) on a real "phone" clip; it detected
   a "cat," nothing resembling a phone. This needs a different/fine-tuned model,
   not a threshold change. **Not fixed this session** — flagged as follow-up.
2. **Sleeping/Writing are unreachable** for these tightly-cropped clips — both
   require `_head_state_from_kps` to return `"down"`, which needs shoulder
   keypoints to be confidently visible; in a tight single-person crop they often
   aren't, so state falls to `"unknown"` instead. **Not fixed this session** —
   may behave differently on real wide classroom video where shoulders are more
   reliably visible; untested.
3. **EAR has zero discriminative power for attentiveness.** Across a larger
   156-clip sample: attentive-labeled clips had mean EAR **0.199**, non-attentive
   **0.213** — statistically indistinguishable (even backwards). Real EAR values
   across the whole dataset ranged ~0.17–0.28 and never approached the old 0.40
   "focused" threshold, so `concentration_pct` (and therefore "Attentive") was
   structurally unreachable regardless of threshold tuning.
4. **`mouth_open_pct` (MAR-based) does discriminate well**: attentive clips had
   **median 0.0** (mouth closed almost the whole time), every other category had
   **median ~0.9–0.98** (mouth open almost the whole time).

**Fix applied** (per user's explicit choice — "let's go with mouth"): replaced the
EAR+gaze+emotion-gated `concentration_pct` formula with
`conc_pct = (1 - mouth_open_pct) * 100`. Added a new named constant
`ATTENTIVE_CONC_THRESH = 80.0` (was a hardcoded `50.0` inline) — derived from a
threshold sweep against the 156-clip sample: binary accuracy plateaus at
**~67–70%** for thresholds 60–95, peaking around 80–85; the old default of 50 would
have scored ~60–64%; the previous EAR-based formula scored effectively randomly
(0% Attentive recall — it was never even predicted).

**Out-of-sample confirmation**: 8 fresh clips never used in calibration — mostly
correct (distracted→Distracted, head_side→Talking x2), one miss (a head_down clip
predicted Attentive), consistent with the measured ~67–70% ceiling rather than a
broken pipeline.

**Honest bottom line**: real, measurable progress (from "Attentive never predicted
at all" to "~2 out of 3 correct on the binary distinction"), not a solved problem.
The mouth-open signal alone has a ceiling. Next steps would likely need either a
genuinely different/combined signal, or more labeled data to see if
mouth+motion+sideways beats mouth alone.

---

## 6. Metrics quick-reference

| Metric | Value | Source |
|---|---|---|
| Attendance similarity threshold (AdaFace) | 0.28 | `eval/impostor_scope_eval.py` p99, clean labeled set |
| Attendance old threshold (glintr100) | 0.38 | pre-existing |
| Classroom within-video re-ID threshold (AdaFace) | 0.35 | judgment call, conservative between roster-threshold and old glintr100 value |
| Classroom old re-ID threshold (glintr100) | 0.40 | pre-existing |
| Roster-match threshold (video recognition, `utils/roster_match.py`) | 0.35 + 0.05 margin | raised from 0.28 after observing false accepts |
| `ATTENTIVE_CONC_THRESH` (mouth-closed %) | 80.0 | threshold sweep, n=156 labeled clips, peak accuracy region |
| EAR mean, attentive vs. non-attentive | 0.199 vs 0.213 | n=156, effectively no separation |
| `mouth_open_pct` median, attentive vs. non-attentive | 0.0 vs ~0.957 | n=156, strong separation |
| Binary attentive/not accuracy, mouth-based | ~67–70% | n=156, threshold sweep |
| Binary attentive/not accuracy, old EAR-based | ~0% recall on Attentive | never predicted at all |
| Phone detection rate on real phone clips | 0% | n=12+ tested, generic COCO model |
| Demo image false-positive-free candidates found | 65 of 216 scanned | clean rerun after gallery-contamination found |
| JAI enrollment: sampled frames before/after sampling fix | 9 → 30 | duration-independent sampling |

---

## 7. Open items for next session

1. **Phone detector is non-functional** — needs a different or fine-tuned model.
2. **Sleeping/Writing detection** untested on real wide classroom video (only
   tested on tightly-cropped single-person clips where it structurally can't fire).
3. **Attentive classification ceiling (~67–70%)** — consider combining signals or
   gathering more labeled data.
4. **Attendance gallery-growth contamination risk** (§1) — deferred at user's
   request, not fixed.
5. **Decide when/whether to push** all local-only work (§2–§5) to `origin`/`hf`.
6. A full 300-clip calibration run was started but crashed around 205/300 (cause
   not investigated — possibly unrelated to this codebase); 156 valid clips were
   still recovered and used. Could rerun for a larger sample if useful.
