// ── Tab switching ─────────────────────────────────────────────────────────────
const tabButtons = Array.from(document.querySelectorAll(".tab-button"));
const tabPanels  = Array.from(document.querySelectorAll(".tab-panel"));

function activateTab(tabId) {
  tabButtons.forEach((btn) => {
    const active = btn.dataset.tabTarget === tabId;
    btn.classList.toggle("active", active);
    btn.setAttribute("aria-selected", String(active));
  });
  tabPanels.forEach((panel) => panel.classList.toggle("hidden", panel.id !== tabId));
}
tabButtons.forEach((btn) => btn.addEventListener("click", () => activateTab(btn.dataset.tabTarget)));

// ── Classroom tab (ClassroomPipeline → /api/classroom/process) ───────────────
const classroomForm      = document.getElementById("classroom-form");
const classroomFileInput = document.getElementById("classroom-video-input");
const classroomFileLabel = document.getElementById("classroom-file-label");
const classroomStatus    = document.getElementById("classroom-status");
const classroomResults   = document.getElementById("classroom-results");
const classroomMetrics   = document.getElementById("classroom-metrics");
const classroomSummaryLink = document.getElementById("classroom-summary-link");
const classroomCsvLink   = document.getElementById("classroom-csv-link");
const classroomClassBar  = document.getElementById("classroom-class-bar");
const classroomStudents  = document.getElementById("classroom-students");

function selectedFileText(files, fallback) {
  if (!files || !files.length) return fallback;
  return files.length === 1 ? files[0].name : `${files[0].name} + ${files.length - 1} more`;
}

classroomFileInput.addEventListener("change", () => {
  classroomFileLabel.textContent = selectedFileText(classroomFileInput.files, "Choose a classroom video (up to 60 min)");
});

const ACTION_COLORS = {
  "Attentive": "#22c55e", "Writing": "#3b82f6", "Talking": "#f59e0b",
  "On Phone":  "#ef4444", "Sleeping": "#8b5cf6", "Distracted": "#f97316",
};
function actionColor(a) { return ACTION_COLORS[a] || "#94a3b8"; }

classroomForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!classroomFileInput.files.length) {
    classroomStatus.textContent = "Choose a video file first.";
    classroomStatus.classList.add("error");
    return;
  }
  const btn = classroomForm.querySelector("button[type='submit']");
  const payload = new FormData();
  payload.append("video", classroomFileInput.files[0]);

  classroomResults.classList.add("hidden");
  classroomStatus.classList.remove("error");
  classroomStatus.textContent = "Analysing classroom — this can take several minutes for long videos...";
  btn.disabled = true;

  try {
    const resp = await fetch("/api/classroom/process", { method: "POST", body: payload });
    const data = await resp.json();
    if (!data.ok) throw new Error(data.error || "Unknown error");

    const s = data.summary;
    classroomStatus.textContent =
      `Done — ${s.student_count} students across ${s.total_windows} windows (${s.duration_seconds}s).`;

    classroomMetrics.innerHTML = [
      ["Students", s.student_count], ["Windows", s.total_windows],
      ["Attentive", `${s.class_attentive_pct}%`], ["Duration", `${Math.round(s.duration_seconds / 60)}m`],
    ].map(([l, v]) => `<div class="metric"><span class="label">${l}</span><span class="value">${v ?? "-"}</span></div>`).join("");

    classroomSummaryLink.href = data.download_urls.summary_json;
    classroomCsvLink.href     = data.download_urls.csv;

    const actionTotals = {}; let totalObs = 0;
    for (const student of (s.students || []))
      for (const win of (student.timeline || []))
        { actionTotals[win.action] = (actionTotals[win.action] || 0) + 1; totalObs++; }

    const barSegs = Object.entries(actionTotals).sort((a, b) => b[1] - a[1])
      .map(([action, count]) => {
        const pct = totalObs ? (count / totalObs * 100).toFixed(1) : 0;
        return `<div class="cls-bar-seg" style="flex:${count};background:${actionColor(action)}" title="${action}: ${pct}%"><span>${action} ${pct}%</span></div>`;
      }).join("");
    classroomClassBar.innerHTML = `<p class="cls-bar-label">Class-wide action distribution</p><div class="cls-bar">${barSegs}</div>`;

    renderClassroomStudents(s.students || []);
    classroomResults.classList.remove("hidden");
  } catch (err) {
    classroomStatus.textContent = `Error: ${err.message}`;
    classroomStatus.classList.add("error");
  } finally { btn.disabled = false; }
});

function renderClassroomStudents(students) {
  if (!students.length) { classroomStudents.innerHTML = "<p class='empty-state'>No students detected.</p>"; return; }
  classroomStudents.innerHTML = students.map((student) => {
    const breakdownBars = Object.entries(student.action_breakdown || {}).sort((a, b) => b[1] - a[1])
      .map(([action, pct]) => `<div class="cls-mini-seg" style="flex:${pct};background:${actionColor(action)}" title="${action}: ${pct}%"></div>`).join("");

    const timelineHtml = (student.timeline || []).map((win) => {
      const mm = Math.floor(win.window_start_seconds / 60).toString().padStart(2, "0");
      const ss = Math.floor(win.window_start_seconds % 60).toString().padStart(2, "0");
      return `<div class="cls-win" style="border-color:${actionColor(win.action)}">
        <div class="cls-win-header" style="background:${actionColor(win.action)}22">
          <span class="cls-win-time">${mm}:${ss}</span>
          <span class="cls-win-action" style="color:${actionColor(win.action)}">${win.action}</span>
          <span class="cls-win-meta">${win.emotion || ""} · ${(win.concentration_pct || 0).toFixed(0)}% conc</span>
        </div>
        ${win.clip_url
          ? `<video class="cls-win-clip" src="${win.clip_url}" controls preload="none" muted playsinline></video>`
          : `<div class="cls-win-no-clip">no clip</div>`}
      </div>`;
    }).join("");

    const attColor = student.attentive_pct >= 70 ? "#22c55e" : student.attentive_pct >= 40 ? "#f59e0b" : "#ef4444";
    const idLabel = student.recognized_name
      ? `<span class="cls-student-name">${student.recognized_name}</span><span class="cls-student-id cls-student-id-secondary">${student.student_label}</span>`
      : `<span class="cls-student-id">${student.student_label}</span>`;
    return `<details class="cls-student-card" open>
      <summary class="cls-student-summary">
        ${idLabel}
        <span class="cls-student-dominant">${student.dominant_action}</span>
        <span class="cls-student-attn" style="color:${attColor}">${student.attentive_pct}% attentive</span>
        <span class="cls-student-windows">${student.windows_seen} windows</span>
        <div class="cls-mini-bar">${breakdownBars}</div>
      </summary>
      <div class="cls-timeline">${timelineHtml}</div>
    </details>`;
  }).join("");
}

// ── Attendance tab ─────────────────────────────────────────────────────────────
const enrollForm        = document.getElementById("enroll-form");
const studentNameInput  = document.getElementById("student-name-input");
const enrollMediaInput  = document.getElementById("enroll-media-input");
const enrollMediaLabel  = document.getElementById("enroll-media-label");
const enrollStatus      = document.getElementById("enroll-status");
const enrollResult      = document.getElementById("enroll-result");

const markForm          = document.getElementById("mark-form");
const classroomPhotoInput = document.getElementById("classroom-photo-input");
const classroomPhotoLabel = document.getElementById("classroom-photo-label");
const markStatus        = document.getElementById("mark-status");
const markResult        = document.getElementById("mark-result");
const markedPhotoPreview= document.getElementById("marked-photo-preview");
const recognizedList    = document.getElementById("recognized-list");
const attendanceLogList = document.getElementById("attendance-log-list");
const rosterList        = document.getElementById("roster-list");
const attendanceLogSummary = document.getElementById("attendance-log-summary");

enrollMediaInput.addEventListener("change", () => {
  enrollMediaLabel.textContent = selectedFileText(enrollMediaInput.files, "Choose photos or videos for enrollment");
});
classroomPhotoInput.addEventListener("change", () => {
  classroomPhotoLabel.textContent = selectedFileText(classroomPhotoInput.files, "Choose a classroom photo");
});

// Enrollment tab toggle
let enrollTab = "files";
function switchEnrollTab(tab) {
  enrollTab = tab;
  document.getElementById("enroll-tab-files").style.display  = tab === "files"  ? "" : "none";
  document.getElementById("enroll-tab-folder").style.display = tab === "folder" ? "" : "none";
  document.getElementById("tab-files").classList.toggle("tab-active",  tab === "files");
  document.getElementById("tab-folder").classList.toggle("tab-active", tab === "folder");
}

enrollForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const studentName = studentNameInput.value.trim();
  if (!studentName) { enrollStatus.textContent = "Enter a student name."; enrollStatus.classList.add("error"); return; }

  const btn = enrollForm.querySelector("button[type='submit']");
  enrollStatus.classList.remove("error");
  enrollStatus.textContent = "Extracting embeddings and saving the student...";
  btn.disabled = true;

  try {
    let response, data;
    if (enrollTab === "folder") {
      const folderPath = document.getElementById("enroll-folder-input").value.trim();
      if (!folderPath) { enrollStatus.textContent = "Enter a folder path."; enrollStatus.classList.add("error"); btn.disabled = false; return; }
      response = await fetch("/api/attendance/enroll-folder", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ student_name: studentName, folder_path: folderPath }),
      });
      data = await response.json();
      if (data.ok) enrollStatus.textContent = `Enrolled ${data.student.name} from ${data.files_used} file(s).`;
    } else {
      if (!enrollMediaInput.files.length) { enrollStatus.textContent = "Upload at least one photo or video."; enrollStatus.classList.add("error"); btn.disabled = false; return; }
      const payload = new FormData();
      payload.append("student_name", studentName);
      Array.from(enrollMediaInput.files).forEach((f) => payload.append("media", f));
      response = await fetch("/api/attendance/enroll", { method: "POST", body: payload });
      data = await response.json();
      if (data.ok) enrollStatus.textContent = `Enrolled ${data.student.name} successfully.`;
    }
    if (!response.ok || !data.ok) throw new Error(data.error || "Enrollment failed.");
    renderRoster(data.students || []);
    renderEnrollmentResult(data.student, data.media_samples || []);
    await refreshAttendanceSummary();
  } catch (err) {
    enrollStatus.textContent = err.message;
    enrollStatus.classList.add("error");
  } finally { btn.disabled = false; }
});

markForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!classroomPhotoInput.files.length) { markStatus.textContent = "Upload a classroom photo first."; markStatus.classList.add("error"); return; }
  const btn = markForm.querySelector("button[type='submit']");
  const payload = new FormData();
  payload.append("photo", classroomPhotoInput.files[0]);
  markStatus.classList.remove("error");
  markStatus.textContent = "Detecting faces and marking attendance...";
  btn.disabled = true;
  try {
    const response = await fetch("/api/attendance/mark", { method: "POST", body: payload });
    const data = await response.json();
    if (!response.ok || !data.ok) throw new Error(data.error || "Attendance marking failed.");
    renderMarkedPhoto(data.marked_url);
    renderRecognizedFaces(data.recognized || [], data.unknown_faces || 0, data.unknown_faces_detail || []);
    renderAttendanceLog(data.attendance_log || []);
    renderRoster(data.roster || []);
    markStatus.textContent = `Marked ${data.recognized.length} student${data.recognized.length === 1 ? "" : "s"}.`;
    markResult.classList.remove("hidden");
    await refreshAttendanceSummary();
  } catch (err) {
    markStatus.textContent = err.message;
    markStatus.classList.add("error");
  } finally { btn.disabled = false; }
});

document.getElementById("demo-preview-btn").addEventListener("click", () => {
  markStatus.classList.remove("error");
  markStatus.textContent = "Demo classroom photo — original, no annotations.";
  markedPhotoPreview.src = "/static/demo_classroom.jpg";
  markResult.classList.remove("hidden");
  recognizedList.innerHTML = "";
  attendanceLogList.innerHTML = "";
  hideUnknownFacesUI();
});

document.getElementById("demo-btn").addEventListener("click", async () => {
  const btn = document.getElementById("demo-btn");
  btn.disabled = true;
  markStatus.classList.remove("error");

  // Step 1 — show original unannotated image immediately
  markResult.classList.remove("hidden");
  markedPhotoPreview.src = "/static/demo_classroom.jpg";
  markStatus.textContent = "Here's the demo classroom photo. Running attendance pipeline...";

  // Step 2 — run the pipeline
  try {
    const response = await fetch("/api/attendance/demo", { method: "POST" });
    const data = await response.json();
    if (!response.ok || !data.ok) throw new Error(data.error || "Demo failed.");
    renderMarkedPhoto(data.marked_url);
    renderRecognizedFaces(data.recognized || [], data.unknown_faces || 0, data.unknown_faces_detail || []);
    renderAttendanceLog(data.attendance_log || []);
    renderRoster(data.roster || []);
    markStatus.textContent = `Demo complete — ${data.recognized.length} student${data.recognized.length === 1 ? "" : "s"} recognized.`;
    await refreshAttendanceSummary();
  } catch (err) {
    markStatus.textContent = err.message;
    markStatus.classList.add("error");
  } finally { btn.disabled = false; }
});

async function refreshAttendanceSummary() {
  try {
    const response = await fetch("/api/attendance/roster");
    const data = await response.json();
    if (response.ok && data.ok) { renderRoster(data.students || []); renderAttendanceSummary(data.attendance || []); }
  } catch (e) { console.error(e); }
}

function renderEnrollmentResult(student, mediaSamples) {
  if (!student) { enrollResult.classList.add("hidden"); return; }
  enrollResult.classList.remove("hidden");
  enrollResult.innerHTML = `
    <div class="result-summary">${student.name} enrolled — ${student.observations ?? 0} embeddings</div>
    <div class="result-detail">${mediaSamples.map((s) => `${s.file_name} (${s.frame_samples} frames)`).join(", ")}</div>`;
}

function renderMarkedPhoto(url) {
  if (!url) { markResult.classList.add("hidden"); return; }
  markResult.classList.remove("hidden");
  markedPhotoPreview.src = url;
}

const unknownFacesToggle = document.getElementById("unknown-faces-toggle");
const unknownFacesGrid   = document.getElementById("unknown-faces-grid");
let currentUnknownFaces  = [];
let unknownFacesExpanded = false;

function renderRecognizedFaces(recognized, unknownFaces, unknownFacesDetail) {
  const names = recognized.map((e) => e.student.name);
  const summary = names.length
    ? `<div class="result-summary">${names.length} present: ${names.join(", ")}</div>`
    : `<div class="result-summary muted">0 present</div>`;

  recognizedList.innerHTML = `<h3>Recognized faces</h3>
    ${summary}
    ${recognized.length
      ? recognized.map((e) => `<div class="result-item"><strong>${e.student.name}</strong><span>Confidence ${formatNumber(e.confidence)}</span></div>`).join("")
      : '<div class="result-item muted">No enrolled students recognized.</div>'}`;

  currentUnknownFaces = unknownFacesDetail || [];
  unknownFacesExpanded = false;
  unknownFacesGrid.classList.add("hidden");
  unknownFacesGrid.innerHTML = "";

  if (currentUnknownFaces.length) {
    unknownFacesToggle.classList.remove("hidden");
    unknownFacesToggle.textContent = `Show unknown faces (${currentUnknownFaces.length})`;
  } else {
    unknownFacesToggle.classList.add("hidden");
  }
}

function hideUnknownFacesUI() {
  currentUnknownFaces = [];
  unknownFacesExpanded = false;
  unknownFacesToggle.classList.add("hidden");
  unknownFacesGrid.classList.add("hidden");
  unknownFacesGrid.innerHTML = "";
}

function cropUnknownFaceThumbnails(imgEl, faces, pad = 26, outSize = 220) {
  const source = document.createElement("canvas");
  source.width = imgEl.naturalWidth;
  source.height = imgEl.naturalHeight;
  const sctx = source.getContext("2d");
  sctx.drawImage(imgEl, 0, 0);

  return faces.map(({ bbox, similarity }) => {
    const [x1, y1, x2, y2] = bbox;
    const px1 = Math.max(0, x1 - pad);
    const py1 = Math.max(0, y1 - pad);
    const px2 = Math.min(source.width, x2 + pad);
    const py2 = Math.min(source.height, y2 + pad);
    const pw = Math.max(1, px2 - px1);
    const ph = Math.max(1, py2 - py1);

    const out = document.createElement("canvas");
    const scale = Math.max(outSize / pw, outSize / ph);
    out.width = Math.round(pw * scale);
    out.height = Math.round(ph * scale);
    const octx = out.getContext("2d");
    octx.imageSmoothingQuality = "high";
    octx.drawImage(source, px1, py1, pw, ph, 0, 0, out.width, out.height);
    return { dataUrl: out.toDataURL("image/jpeg", 0.88), similarity };
  });
}

function renderUnknownFacesGrid() {
  const thumbs = cropUnknownFaceThumbnails(markedPhotoPreview, currentUnknownFaces);
  unknownFacesGrid.innerHTML = thumbs
    .map(
      (t) => `
        <figure class="unknown-face-card">
          <img src="${t.dataUrl}" alt="Unrecognized face, similarity ${formatNumber(t.similarity)}" />
          <figcaption>Unknown &middot; ${formatNumber(t.similarity)}</figcaption>
        </figure>`
    )
    .join("");
}

unknownFacesToggle.addEventListener("click", () => {
  unknownFacesExpanded = !unknownFacesExpanded;
  if (unknownFacesExpanded) {
    const build = () => renderUnknownFacesGrid();
    if (markedPhotoPreview.complete && markedPhotoPreview.naturalWidth) build();
    else markedPhotoPreview.addEventListener("load", build, { once: true });
    unknownFacesGrid.classList.remove("hidden");
    unknownFacesToggle.textContent = `Hide unknown faces (${currentUnknownFaces.length})`;
  } else {
    unknownFacesGrid.classList.add("hidden");
    unknownFacesToggle.textContent = `Show unknown faces (${currentUnknownFaces.length})`;
  }
});

function renderAttendanceLog(log) {
  attendanceLogList.innerHTML = `<h3>Attendance log</h3>
    ${log.map((e) => `<div class="result-item"><strong>${e.student_name}</strong><span>${e.recognized_at} · ${formatNumber(e.confidence)}</span></div>`).join("")}`;
}

function renderAttendanceSummary(log) {
  attendanceLogSummary.innerHTML = `<h3>Recent attendance</h3>
    ${log.map((e) => `<div class="result-item"><strong>${e.student_name}</strong><span>${e.recognized_at} · ${formatNumber(e.confidence)}</span></div>`).join("")}`;
}

function renderRoster(students) {
  if (!students.length) { rosterList.innerHTML = '<div class="result-item muted">No students enrolled yet.</div>'; return; }
  rosterList.innerHTML = students.map((s) => `
    <div class="roster-item" data-student-id="${s.student_id}">
      <div class="roster-item-main">
        <strong>${s.name}</strong>
        <span>${s.observations ?? 0} embeddings · ${s.updated_at ?? "-"}</span>
      </div>
      <button class="delete-student-button" type="button" data-student-id="${s.student_id}">Delete</button>
    </div>`).join("");
}

rosterList.addEventListener("click", async (event) => {
  const btn = event.target.closest(".delete-student-button");
  if (!btn) return;
  const studentId = btn.dataset.studentId;
  const name = btn.closest(".roster-item")?.querySelector("strong")?.textContent || "this student";
  if (!confirm(`Delete ${name}? This removes the student and their attendance records.`)) return;
  btn.disabled = true; btn.textContent = "Deleting...";
  try {
    const response = await fetch(`/api/attendance/students/${encodeURIComponent(studentId)}`, { method: "DELETE" });
    const data = await response.json();
    if (!response.ok || !data.ok) throw new Error(data.error || "Delete failed.");
    renderRoster(data.students || []);
    renderAttendanceSummary(data.attendance || []);
    await refreshAttendanceSummary();
  } catch (err) { alert(err.message); }
  finally { btn.disabled = false; btn.textContent = "Delete"; }
});

// ── Lightbox ──────────────────────────────────────────────────────────────────
const lightbox      = document.getElementById("photo-lightbox");
const lightboxImg   = document.getElementById("lightbox-img");
const lightboxClose = document.getElementById("lightbox-close");
markedPhotoPreview.addEventListener("click", () => { lightboxImg.src = markedPhotoPreview.src; lightbox.classList.remove("hidden"); });
lightboxClose.addEventListener("click", () => lightbox.classList.add("hidden"));
lightbox.addEventListener("click", (e) => { if (e.target === lightbox) lightbox.classList.add("hidden"); });
document.addEventListener("keydown", (e) => { if (e.key === "Escape") lightbox.classList.add("hidden"); });

// ── Helpers ───────────────────────────────────────────────────────────────────
function formatWindow(s) { const n = Number(s); return isNaN(n) ? "-" : `${n.toFixed(2)}s`; }
function formatNumber(v) { if (v == null || isNaN(Number(v))) return "-"; return Number(v).toFixed(4); }

refreshAttendanceSummary();
