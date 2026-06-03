const tabButtons = Array.from(document.querySelectorAll(".tab-button"));
const tabPanels = Array.from(document.querySelectorAll(".tab-panel"));

const activityForm = document.getElementById("upload-form");
const videoInput = document.getElementById("video-input");
const fileLabel = document.getElementById("file-label");
const statusBox = document.getElementById("status");
const resultsBox = document.getElementById("results");
const metricsBox = document.getElementById("metrics");
const studentGroupsBox = document.getElementById("student-groups");
const emptyStateBox = document.getElementById("empty-state");
const summaryLink = document.getElementById("summary-link");
const csvLink = document.getElementById("csv-link");
const annotatedLink = document.getElementById("annotated-link");

const enrollForm = document.getElementById("enroll-form");
const studentNameInput = document.getElementById("student-name-input");
const enrollMediaInput = document.getElementById("enroll-media-input");
const enrollMediaLabel = document.getElementById("enroll-media-label");
const enrollStatus = document.getElementById("enroll-status");
const enrollResult = document.getElementById("enroll-result");

const markForm = document.getElementById("mark-form");
const classroomPhotoInput = document.getElementById("classroom-photo-input");
const classroomPhotoLabel = document.getElementById("classroom-photo-label");
const markStatus = document.getElementById("mark-status");
const markResult = document.getElementById("mark-result");
const markedPhotoPreview = document.getElementById("marked-photo-preview");
const recognizedList = document.getElementById("recognized-list");
const attendanceLogList = document.getElementById("attendance-log-list");
const rosterList = document.getElementById("roster-list");
const attendanceLogSummary = document.getElementById("attendance-log-summary");

function activateTab(tabId) {
  tabButtons.forEach((button) => {
    const isActive = button.dataset.tabTarget === tabId;
    button.classList.toggle("active", isActive);
    button.setAttribute("aria-selected", String(isActive));
  });

  tabPanels.forEach((panel) => {
    panel.classList.toggle("hidden", panel.id !== tabId);
  });
}

tabButtons.forEach((button) => {
  button.addEventListener("click", () => activateTab(button.dataset.tabTarget));
});

function selectedFileText(files, fallbackText) {
  if (!files || files.length === 0) {
    return fallbackText;
  }
  if (files.length === 1) {
    return files[0].name;
  }
  return `${files[0].name} + ${files.length - 1} more`;
}

videoInput.addEventListener("change", () => {
  fileLabel.textContent = selectedFileText(videoInput.files, "Choose a video file or drag one here");
});

enrollMediaInput.addEventListener("change", () => {
  enrollMediaLabel.textContent = selectedFileText(enrollMediaInput.files, "Choose photos or a video for enrollment");
});

classroomPhotoInput.addEventListener("change", () => {
  classroomPhotoLabel.textContent = selectedFileText(classroomPhotoInput.files, "Choose a classroom photo");
});

activityForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  if (!videoInput.files.length) {
    statusBox.textContent = "Choose a video file first.";
    statusBox.classList.add("error");
    return;
  }

  const submitButton = activityForm.querySelector("button[type='submit']");
  const payload = new FormData();
  payload.append("video", videoInput.files[0]);

  resultsBox.classList.add("hidden");
  statusBox.classList.remove("error");
  statusBox.textContent = "Processing video. This can take a while...";
  submitButton.disabled = true;

  try {
    const response = await fetch("/api/process", {
      method: "POST",
      body: payload,
    });
    const data = await response.json();

    if (!response.ok || !data.ok) {
      throw new Error(data.error || "Processing failed.");
    }

    renderMetrics(data.summary, data.job_id);
    renderLinks(data.download_urls);
    renderGroupedStudents(data.clips || []);
    statusBox.textContent = "Done. Review the summary below.";
    resultsBox.classList.remove("hidden");
  } catch (error) {
    statusBox.textContent = error.message;
    statusBox.classList.add("error");
  } finally {
    submitButton.disabled = false;
  }
});

// Tab switching for enrollment
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
  if (!studentName) {
    enrollStatus.textContent = "Enter a student name.";
    enrollStatus.classList.add("error");
    return;
  }

  const submitButton = enrollForm.querySelector("button[type='submit']");
  enrollStatus.classList.remove("error");
  enrollStatus.textContent = "Extracting embeddings and saving the student...";
  submitButton.disabled = true;

  try {
    let response, data;

    if (enrollTab === "folder") {
      const folderPath = document.getElementById("enroll-folder-input").value.trim();
      if (!folderPath) {
        enrollStatus.textContent = "Enter a folder path.";
        enrollStatus.classList.add("error");
        submitButton.disabled = false;
        return;
      }
      response = await fetch("/api/attendance/enroll-folder", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ student_name: studentName, folder_path: folderPath }),
      });
      data = await response.json();
      if (data.ok) {
        enrollStatus.textContent =
          `Enrolled ${data.student.name} from ${data.files_used} file(s) successfully.`;
      }
    } else {
      if (!enrollMediaInput.files.length) {
        enrollStatus.textContent = "Upload one or more photos or a video.";
        enrollStatus.classList.add("error");
        submitButton.disabled = false;
        return;
      }
      const payload = new FormData();
      payload.append("student_name", studentName);
      Array.from(enrollMediaInput.files).forEach((file) => payload.append("media", file));
      response = await fetch("/api/attendance/enroll", { method: "POST", body: payload });
      data = await response.json();
      if (data.ok) {
        enrollStatus.textContent = `Enrolled ${data.student.name} successfully.`;
      }
    }

    if (!response.ok || !data.ok) {
      throw new Error(data.error || "Enrollment failed.");
    }

    renderRoster(data.students || []);
    renderEnrollmentResult(data.student, data.media_samples || []);
    await refreshAttendanceSummary();
  } catch (error) {
    enrollStatus.textContent = error.message;
    enrollStatus.classList.add("error");
  } finally {
    submitButton.disabled = false;
  }
});

markForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  if (!classroomPhotoInput.files.length) {
    markStatus.textContent = "Upload a classroom photo first.";
    markStatus.classList.add("error");
    return;
  }

  const submitButton = markForm.querySelector("button[type='submit']");
  const payload = new FormData();
  payload.append("photo", classroomPhotoInput.files[0]);

  markStatus.classList.remove("error");
  markStatus.textContent = "Detecting faces and marking attendance...";
  submitButton.disabled = true;

  try {
    const response = await fetch("/api/attendance/mark", {
      method: "POST",
      body: payload,
    });
    const data = await response.json();

    if (!response.ok || !data.ok) {
      throw new Error(data.error || "Attendance marking failed.");
    }

    renderMarkedPhoto(data.marked_url);
    renderRecognizedFaces(data.recognized || [], data.unknown_faces || 0);
    renderAttendanceLog(data.attendance_log || []);
    renderRoster(data.roster || []);
    markStatus.textContent = `Marked attendance for ${data.recognized.length} student${data.recognized.length === 1 ? "" : "s"}.`;
    markResult.classList.remove("hidden");
    await refreshAttendanceSummary();
  } catch (error) {
    markStatus.textContent = error.message;
    markStatus.classList.add("error");
  } finally {
    submitButton.disabled = false;
  }
});

async function refreshAttendanceSummary() {
  try {
    const response = await fetch("/api/attendance/roster");
    const data = await response.json();
    if (response.ok && data.ok) {
      renderRoster(data.students || []);
      renderAttendanceSummary(data.attendance || []);
    }
  } catch (error) {
    console.error(error);
  }
}

function renderMetrics(summary, jobId) {
  const items = [
    ["Job", jobId],
    ["Video", summary.video_name],
    ["Clips", summary.clip_count],
    ["Students", summary.student_count],
    ["Frames", summary.total_frames],
  ];

  metricsBox.innerHTML = items
    .map(([label, value]) => `<div class="metric"><span class="label">${label}</span><span class="value">${value ?? "-"}</span></div>`)
    .join("");
}

function renderLinks(downloadUrls) {
  summaryLink.href = downloadUrls.summary_json || "#";
  csvLink.href = downloadUrls.csv || "#";
  annotatedLink.href = downloadUrls.annotated_video || "#";
}

function renderGroupedStudents(clips) {
  const grouped = groupClipsByStudent(clips);

  if (grouped.length === 0) {
    studentGroupsBox.innerHTML = "";
    emptyStateBox.classList.remove("hidden");
    return;
  }

  emptyStateBox.classList.add("hidden");
  studentGroupsBox.innerHTML = grouped
    .map(
      ({ studentLabel, clips: studentClips }) => `
        <section class="student-card">
          <div class="student-card-header">
            <div>
              <p class="student-title">${studentLabel}</p>
              <p class="student-meta">${studentClips.length} engagement clip${studentClips.length === 1 ? "" : "s"}</p>
            </div>
          </div>
          <div class="clip-gallery">
            ${studentClips
              .map(
                (clip) => `
                  <article class="clip-card">
                    <video class="clip-player" controls preload="metadata" src="${clip.clip_url || ""}"></video>
                    <div class="clip-card-body">
                      <div class="clip-badges">
                        <span class="clip-badge ${clip.decision_source === "phone_detector" ? "clip-badge-phone" : "clip-badge-cnn"}">${clip.decision_source || "3dcnn"}</span>
                        <span class="clip-badge">${formatWindow(clip.window_start_seconds)}</span>
                      </div>
                      <p class="clip-summary">${clip.predicted_label ?? "-"} · confidence ${formatNumber(clip.confidence)}</p>
                      <p class="clip-detail">Low ${formatNumber(clip.low_engagement)} · High ${formatNumber(clip.high_engagement)}</p>
                      <a class="clip-link" href="${clip.clip_url || "#"}" target="_blank" rel="noreferrer">Open clip</a>
                    </div>
                  </article>
                `,
              )
              .join("")}
          </div>
          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Window</th>
                  <th>Label</th>
                  <th>Confidence</th>
                  <th>Low</th>
                  <th>High</th>
                  <th>Decision</th>
                </tr>
              </thead>
              <tbody>
                ${studentClips
                  .map(
                    (clip) => `
                      <tr>
                        <td>${formatWindow(clip.window_start_seconds)}</td>
                        <td>${clip.predicted_label ?? "-"}</td>
                        <td>${formatNumber(clip.confidence)}</td>
                        <td>${formatNumber(clip.low_engagement)}</td>
                        <td>${formatNumber(clip.high_engagement)}</td>
                        <td>${clip.decision_source || "3dcnn"}</td>
                      </tr>
                    `,
                  )
                  .join("")}
              </tbody>
            </table>
          </div>
        </section>
      `,
    )
    .join("");
}

function groupClipsByStudent(clips) {
  const bucket = new Map();

  clips
    .slice()
    .sort((left, right) => {
      const leftStudent = studentSortKey(left);
      const rightStudent = studentSortKey(right);
      if (leftStudent !== rightStudent) {
        return leftStudent.localeCompare(rightStudent, undefined, { numeric: true, sensitivity: "base" });
      }
      return Number(left.window_start_seconds ?? 0) - Number(right.window_start_seconds ?? 0);
    })
    .forEach((clip) => {
      const key = studentSortKey(clip);
      if (!bucket.has(key)) {
        bucket.set(key, []);
      }
      bucket.get(key).push(clip);
    });

  return Array.from(bucket.entries()).map(([studentLabel, studentClips]) => ({ studentLabel, clips: studentClips }));
}

function studentSortKey(clip) {
  if (clip.student_label) {
    return clip.student_label;
  }
  if (clip.student_id !== null && clip.student_id !== undefined) {
    return `student_${String(clip.student_id).padStart(3, "0")}`;
  }
  return "Unknown student";
}

function renderEnrollmentResult(student, mediaSamples) {
  if (!student) {
    enrollResult.classList.add("hidden");
    return;
  }

  enrollResult.classList.remove("hidden");
  enrollResult.innerHTML = `
    <div class="result-summary">${student.name} enrolled successfully.</div>
    <div class="result-detail">Observations: ${student.observations ?? 0}</div>
    <div class="result-detail">Media samples: ${mediaSamples.map((sample) => `${sample.file_name} (${sample.frame_samples})`).join(", ")}</div>
  `;
}

function renderMarkedPhoto(url) {
  if (!url) {
    markResult.classList.add("hidden");
    return;
  }

  markResult.classList.remove("hidden");
  markedPhotoPreview.src = url;
}

// Lightbox
const lightbox      = document.getElementById("photo-lightbox");
const lightboxImg   = document.getElementById("lightbox-img");
const lightboxClose = document.getElementById("lightbox-close");

markedPhotoPreview.addEventListener("click", () => {
  lightboxImg.src = markedPhotoPreview.src;
  lightbox.classList.remove("hidden");
});
lightboxClose.addEventListener("click", () => lightbox.classList.add("hidden"));
lightbox.addEventListener("click", (e) => {
  if (e.target === lightbox) lightbox.classList.add("hidden");
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") lightbox.classList.add("hidden");
});

function renderRecognizedFaces(recognized, unknownFaces) {
  const recognizedRows = recognized.length
    ? recognized
        .map(
          (entry) => `
            <div class="result-item">
              <strong>${entry.student.name}</strong>
              <span>Confidence ${formatNumber(entry.confidence)}</span>
            </div>
          `,
        )
        .join("")
    : '<div class="result-item muted">No enrolled students recognized.</div>';

  recognizedList.innerHTML = `
    <h3>Recognized faces</h3>
    ${recognizedRows}
    <div class="result-item muted">Unknown faces: ${unknownFaces}</div>
  `;
}

function renderAttendanceLog(attendanceLog) {
  attendanceLogList.innerHTML = `
    <h3>Attendance log</h3>
    ${attendanceLog
      .map(
        (entry) => `
          <div class="result-item">
            <strong>${entry.student_name}</strong>
            <span>${entry.recognized_at} · ${entry.source} · ${formatNumber(entry.confidence)}</span>
          </div>
        `,
      )
      .join("")}
  `;
}

function renderAttendanceSummary(attendanceLog) {
  attendanceLogSummary.innerHTML = `
    <h3>Recent attendance activity</h3>
    ${attendanceLog
      .map(
        (entry) => `
          <div class="result-item">
            <strong>${entry.student_name}</strong>
            <span>${entry.recognized_at} · ${formatNumber(entry.confidence)}</span>
          </div>
        `,
      )
      .join("")}
  `;
}

function renderRoster(students) {
  if (!students.length) {
    rosterList.innerHTML = '<div class="result-item muted">No students enrolled yet.</div>';
    return;
  }

  rosterList.innerHTML = students
    .map(
      (student) => `
        <div class="roster-item" data-student-id="${student.student_id}">
          <div class="roster-item-main">
            <strong>${student.name}</strong>
            <span>${student.observations ?? 0} embeddings · updated ${student.updated_at ?? "-"}</span>
          </div>
          <button class="delete-student-button" type="button" data-student-id="${student.student_id}">Delete</button>
        </div>
      `,
    )
    .join("");
}

rosterList.addEventListener("click", async (event) => {
  const button = event.target.closest(".delete-student-button");
  if (!button) {
    return;
  }

  const studentId = button.dataset.studentId;
  if (!studentId) {
    return;
  }

  const studentCard = button.closest(".roster-item");
  const studentName = studentCard?.querySelector("strong")?.textContent || "this student";
  const confirmed = window.confirm(`Delete ${studentName}? This will remove the student and their attendance records.`);
  if (!confirmed) {
    return;
  }

  button.disabled = true;
  button.textContent = "Deleting...";

  try {
    const response = await fetch(`/api/attendance/students/${encodeURIComponent(studentId)}`, {
      method: "DELETE",
    });
    const data = await response.json();

    if (!response.ok || !data.ok) {
      throw new Error(data.error || "Delete failed.");
    }

    renderRoster(data.students || []);
    renderAttendanceSummary(data.attendance || []);
    enrollStatus.classList.remove("error");
    markStatus.classList.remove("error");
    await refreshAttendanceSummary();
  } catch (error) {
    alert(error.message);
  } finally {
    button.disabled = false;
    button.textContent = "Delete";
  }
});

function formatWindow(seconds) {
  const numericSeconds = Number(seconds);
  if (Number.isNaN(numericSeconds)) {
    return "-";
  }
  return `${numericSeconds.toFixed(2)}s`;
}

function formatNumber(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  return Number(value).toFixed(4);
}

refreshAttendanceSummary();

// ── Engagement tab ────────────────────────────────────────────────────────────

const engagementForm        = document.getElementById("engagement-form");
const engagementVideoInput  = document.getElementById("engagement-video-input");
const engagementFileLabel   = document.getElementById("engagement-file-label");
const engagementStatus      = document.getElementById("engagement-status");
const engagementResults     = document.getElementById("engagement-results");
const engagementMetrics     = document.getElementById("engagement-metrics");
const engagementStudents    = document.getElementById("engagement-students");
const engagementSummaryLink = document.getElementById("engagement-summary-link");
const engagementCsvLink     = document.getElementById("engagement-csv-link");

engagementVideoInput.addEventListener("change", () => {
  engagementFileLabel.textContent = selectedFileText(engagementVideoInput.files, "Choose a classroom video");
});

engagementForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!engagementVideoInput.files.length) {
    engagementStatus.textContent = "Choose a video file first.";
    engagementStatus.classList.add("error");
    return;
  }

  const btn = engagementForm.querySelector("button[type='submit']");
  const payload = new FormData();
  payload.append("video", engagementVideoInput.files[0]);

  engagementResults.classList.add("hidden");
  engagementStatus.classList.remove("error");
  engagementStatus.textContent = "Analysing engagement — this can take a few minutes...";
  btn.disabled = true;

  try {
    const response = await fetch("/api/engagement/process", { method: "POST", body: payload });
    const data = await response.json();
    if (!response.ok || !data.ok) throw new Error(data.error || "Processing failed.");

    renderEngagementMetrics(data.summary);
    renderEngagementStudents(data.students || []);
    engagementSummaryLink.href = data.download_urls.summary_json || "#";
    engagementCsvLink.href     = data.download_urls.csv || "#";
    engagementStatus.textContent = `Done. ${data.summary.student_count} students tracked · class score ${data.summary.class_engagement_score}`;
    engagementResults.classList.remove("hidden");
  } catch (err) {
    engagementStatus.textContent = err.message;
    engagementStatus.classList.add("error");
  } finally {
    btn.disabled = false;
  }
});

function renderEngagementMetrics(s) {
  const items = [
    ["Students",    s.student_count],
    ["Class score", s.class_engagement_score],
    ["Attentive",   engPct(s.attentive_pct)],
    ["Writing",     engPct(s.writing_pct)],
    ["Sleeping",    engPct(s.sleeping_pct)],
    ["On phone",    engPct(s.phone_pct)],
    ["Duration",    `${s.duration_seconds}s`],
  ];
  engagementMetrics.innerHTML = items
    .map(([l, v]) => `<div class="metric"><span class="label">${l}</span><span class="value">${v ?? "-"}</span></div>`)
    .join("");
}

const _statusColor = { attentive: "#4caf50", writing: "#2196f3", sleeping: "#f44336", phone: "#ff9800", distracted: "#9c27b0", uncertain: "#888" };

function renderEngagementStudents(students) {
  if (!students.length) {
    engagementStudents.innerHTML = '<p style="color:#888;padding:12px">No students tracked.</p>';
    return;
  }
  engagementStudents.innerHTML = students.map((s) => `
    <section class="student-card">
      <div class="student-card-header">
        <div>
          <p class="student-title">Track #${s.track_id}</p>
          <p class="student-meta" style="color:${_statusColor[s.status] || "#888"}">${s.status} · score ${s.engagement_score}</p>
        </div>
      </div>
      ${(s.clip_urls && s.clip_urls.length) ? `
      <div class="clip-gallery">
        ${s.clip_urls.map((url, i) => `
        <article class="clip-card">
          <video class="clip-player" controls preload="metadata" src="${url}"></video>
          <div class="clip-card-body">
            <div class="clip-badges">
              <span class="clip-badge ${s.status === 'phone' ? 'clip-badge-phone' : 'clip-badge-cnn'}">${s.status}</span>
              <span class="clip-badge">Part ${i + 1}</span>
            </div>
            <a class="clip-link" href="${url}" target="_blank" rel="noreferrer">Open clip</a>
          </div>
        </article>`).join("")}
      </div>` : ""}
      <div class="metrics">
        <div class="metric"><span class="label">Attentive</span><span class="value">${engPct(s.head_forward_pct)}</span></div>
        <div class="metric"><span class="label">Writing</span><span class="value">${engPct(s.writing_pct)}</span></div>
        <div class="metric"><span class="label">Sleeping</span><span class="value">${engPct(s.sleeping_pct)}</span></div>
        <div class="metric"><span class="label">Phone</span><span class="value">${engPct(s.phone_pct)}</span></div>
        <div class="metric"><span class="label">Frames</span><span class="value">${s.frames_tracked}</span></div>
      </div>
    </section>
  `).join("");
}

function engPct(v) {
  if (v === null || v === undefined) return "-";
  return `${Math.round(Number(v) * 100)}%`;
}

// ── Cognitive (Paper) tab ─────────────────────────────────────────────────────

const cogForm        = document.getElementById("cognitive-form");
const cogVideoInput  = document.getElementById("cognitive-video-input");
const cogFileLabel   = document.getElementById("cognitive-file-label");
const cogStatus      = document.getElementById("cognitive-status");
const cogResults     = document.getElementById("cognitive-results");
const cogMetrics     = document.getElementById("cognitive-metrics");
const cogStudents    = document.getElementById("cognitive-students");
const cogSummaryLink = document.getElementById("cognitive-summary-link");
const cogCsvLink     = document.getElementById("cognitive-csv-link");

cogVideoInput.addEventListener("change", () => {
  cogFileLabel.textContent = selectedFileText(cogVideoInput.files, "Choose a classroom video");
});

let _cogBarChart = null;
let _cogPieChart = null;

cogForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!cogVideoInput.files.length) {
    cogStatus.textContent = "Choose a video file first.";
    cogStatus.classList.add("error");
    return;
  }

  const btn = cogForm.querySelector("button[type='submit']");
  const payload = new FormData();
  payload.append("video", cogVideoInput.files[0]);

  cogResults.classList.add("hidden");
  cogStatus.classList.remove("error");
  cogStatus.textContent = "Running cognitive analysis — this may take several minutes…";
  btn.disabled = true;

  try {
    const response = await fetch("/api/cognitive/process", { method: "POST", body: payload });
    const data = await response.json();
    if (!response.ok || !data.ok) throw new Error(data.error || "Processing failed.");

    renderCognitiveMetrics(data.summary);
    renderCognitiveCharts(data.students, data.summary);
    renderCognitiveStudents(data.students);
    cogSummaryLink.href = data.download_urls.summary_json || "#";
    cogCsvLink.href     = data.download_urls.csv || "#";
    const s = data.summary;
    cogStatus.textContent = `Done — ${s.student_count} students · class concentration ${s.class_concentration_pct}% · ${s.present_count} present`;
    cogResults.classList.remove("hidden");
  } catch (err) {
    cogStatus.textContent = err.message;
    cogStatus.classList.add("error");
  } finally {
    btn.disabled = false;
  }
});

function renderCognitiveMetrics(s) {
  const items = [
    ["Students",          s.student_count],
    ["Class conc.",       `${s.class_concentration_pct}%`],
    ["Present",           s.present_count],
    ["Absent",            s.absent_count],
    ["High attention",    s.high_attention_count],
    ["Medium attention",  s.medium_attention_count],
    ["Low attention",     s.low_attention_count],
    ["Duration",          `${s.duration_seconds}s`],
  ];
  cogMetrics.innerHTML = items
    .map(([l, v]) => `<div class="metric"><span class="label">${l}</span><span class="value">${v ?? "-"}</span></div>`)
    .join("");
}

const _EMOTION_COLORS = {
  happy:    "#4caf50",
  neutral:  "#2196f3",
  surprise: "#ff9800",
  sad:      "#9c27b0",
  angry:    "#f44336",
  fear:     "#607d8b",
  disgust:  "#795548",
};

function renderCognitiveCharts(students, summary) {
  // Bar chart — concentration per student
  const barCtx = document.getElementById("cognitive-bar-chart").getContext("2d");
  if (_cogBarChart) _cogBarChart.destroy();
  const barLabels = students.map((s) => s.student_label || `S${s.student_id}`);
  const barValues = students.map((s) => s.concentration_pct);
  const barColors = barValues.map((v) => v >= 70 ? "#4caf50" : v >= 40 ? "#ff9800" : "#f44336");
  _cogBarChart = new Chart(barCtx, {
    type: "bar",
    data: {
      labels: barLabels,
      datasets: [{ label: "Concentration %", data: barValues, backgroundColor: barColors, borderRadius: 4 }],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        y: { min: 0, max: 100, ticks: { callback: (v) => `${v}%` } },
        x: { ticks: { maxRotation: 45 } },
      },
    },
  });

  // Pie chart — class emotion distribution
  const pieCtx = document.getElementById("cognitive-pie-chart").getContext("2d");
  if (_cogPieChart) _cogPieChart.destroy();
  const emotDist = summary.class_emotion_dist || {};
  const pieLabels = Object.keys(emotDist);
  const pieValues = Object.values(emotDist);
  const pieColors = pieLabels.map((k) => _EMOTION_COLORS[k] || "#aaa");
  _cogPieChart = new Chart(pieCtx, {
    type: "pie",
    data: {
      labels: pieLabels.map((l) => l.charAt(0).toUpperCase() + l.slice(1)),
      datasets: [{ data: pieValues, backgroundColor: pieColors, borderWidth: 2 }],
    },
    options: {
      responsive: true,
      plugins: { legend: { position: "right" } },
    },
  });
}

function renderCognitiveStudents(students) {
  if (!students.length) {
    cogStudents.innerHTML = '<p style="color:#888;padding:12px">No students tracked.</p>';
    return;
  }

  const rows = students.map((s, idx) => {
    const attnClass = s.attention_status === "High" ? "attn-high" : s.attention_status === "Medium" ? "attn-medium" : "attn-low";
    const attendClass = s.attendance === "Present" ? "attend-present" : "attend-absent";
    const concPct = Number(s.concentration_pct || 0);
    const concBar = `<div class="conc-bar-bg"><div class="conc-bar-fill" style="width:${Math.min(100, concPct)}%"></div></div>`;
    const clipCell = s.clip_url
      ? `<video class="cog-clip-thumb" controls preload="metadata" src="${s.clip_url}"></video>`
      : `<span style="color:#aaa">—</span>`;
    return `
      <tr>
        <td>${idx + 1}</td>
        <td>${s.student_label || `Student ${s.student_id}`}</td>
        <td>${s.detected_action || "—"}</td>
        <td><span class="attn-badge ${attnClass}">${s.attention_status}</span></td>
        <td>${(s.emotion || "—").charAt(0).toUpperCase() + (s.emotion || "").slice(1)}</td>
        <td>${concPct.toFixed(1)}% ${concBar}</td>
        <td>${s.detected_time_s ?? "—"}s</td>
        <td class="${attendClass}">${s.attendance}</td>
        <td>${s.blink_count ?? "—"}</td>
        <td>${clipCell}</td>
      </tr>`;
  }).join("");

  cogStudents.innerHTML = `
    <div class="cognitive-table-wrap">
      <table class="cognitive-table">
        <thead>
          <tr>
            <th>#</th>
            <th>Student</th>
            <th>Detected Action</th>
            <th>Attention Status</th>
            <th>Emotion</th>
            <th>Concentration</th>
            <th>Detected Time</th>
            <th>Attendance</th>
            <th>Blinks</th>
            <th>Clip</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;
}

// ── Combined tab ──────────────────────────────────────────────────────────────

const combForm        = document.getElementById("combined-form");
const combVideoInput  = document.getElementById("combined-video-input");
const combFileLabel   = document.getElementById("combined-file-label");
const combStatus      = document.getElementById("combined-status");
const combResults     = document.getElementById("combined-results");
const combMetrics     = document.getElementById("combined-metrics");
const combStudents    = document.getElementById("combined-students");
const combSummaryLink = document.getElementById("combined-summary-link");
const combCsvLink     = document.getElementById("combined-csv-link");

combVideoInput.addEventListener("change", () => {
  combFileLabel.textContent = selectedFileText(combVideoInput.files, "Choose a classroom video");
});

let _combBarChart = null;
let _combPieChart = null;

combForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!combVideoInput.files.length) {
    combStatus.textContent = "Choose a video file first.";
    combStatus.classList.add("error");
    return;
  }

  const btn = combForm.querySelector("button[type='submit']");
  const payload = new FormData();
  payload.append("video", combVideoInput.files[0]);

  combResults.classList.add("hidden");
  combStatus.classList.remove("error");
  combStatus.textContent = "Running combined analysis — this may take several minutes…";
  btn.disabled = true;

  try {
    const response = await fetch("/api/combined/process", { method: "POST", body: payload });
    const data = await response.json();
    if (!response.ok || !data.ok) throw new Error(data.error || "Processing failed.");

    renderCombinedMetrics(data.summary);
    renderCombinedCharts(data.students, data.summary);
    renderCombinedStudents(data.students);
    combSummaryLink.href = data.download_urls.summary_json || "#";
    combCsvLink.href     = data.download_urls.csv || "#";
    const s = data.summary;
    combStatus.textContent =
      `Done — ${s.student_count} students · concentration ${s.class_concentration_pct}% · engagement ${s.class_engagement_score}`;
    combResults.classList.remove("hidden");
  } catch (err) {
    combStatus.textContent = err.message;
    combStatus.classList.add("error");
  } finally {
    btn.disabled = false;
  }
});

function renderCombinedMetrics(s) {
  const items = [
    ["Students",       s.student_count],
    ["Concentration",  `${s.class_concentration_pct}%`],
    ["Engagement",     s.class_engagement_score],
    ["Present",        s.present_count],
    ["Absent",         s.absent_count],
    ["High attention", s.high_attention_count],
    ["Med attention",  s.medium_attention_count],
    ["Low attention",  s.low_attention_count],
    ["Attentive",      engPct(s.attentive_pct)],
    ["Writing",        engPct(s.writing_pct)],
    ["Sleeping",       engPct(s.sleeping_pct)],
    ["On phone",       engPct(s.phone_pct)],
    ["Talking",        engPct(s.talking_pct)],
    ["Duration",       `${s.duration_seconds}s`],
  ];
  combMetrics.innerHTML = items
    .map(([l, v]) => `<div class="metric"><span class="label">${l}</span><span class="value">${v ?? "-"}</span></div>`)
    .join("");
}

function renderCombinedCharts(students, summary) {
  const barCtx = document.getElementById("combined-bar-chart").getContext("2d");
  if (_combBarChart) _combBarChart.destroy();
  const barLabels = students.map((s) => s.student_label || `S${s.track_id}`);
  const barValues = students.map((s) => s.concentration_pct);
  const barColors = barValues.map((v) => v >= 70 ? "#4caf50" : v >= 40 ? "#ff9800" : "#f44336");
  _combBarChart = new Chart(barCtx, {
    type: "bar",
    data: {
      labels: barLabels,
      datasets: [{ label: "Concentration %", data: barValues, backgroundColor: barColors, borderRadius: 4 }],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        y: { min: 0, max: 100, ticks: { callback: (v) => `${v}%` } },
        x: { ticks: { maxRotation: 45 } },
      },
    },
  });

  const pieCtx = document.getElementById("combined-pie-chart").getContext("2d");
  if (_combPieChart) _combPieChart.destroy();
  const emotDist = summary.class_emotion_dist || {};
  const pieLabels = Object.keys(emotDist);
  const pieValues = Object.values(emotDist);
  const pieColors = pieLabels.map((k) => _EMOTION_COLORS[k] || "#aaa");
  _combPieChart = new Chart(pieCtx, {
    type: "pie",
    data: {
      labels: pieLabels.map((l) => l.charAt(0).toUpperCase() + l.slice(1)),
      datasets: [{ data: pieValues, backgroundColor: pieColors, borderWidth: 2 }],
    },
    options: { responsive: true, plugins: { legend: { position: "right" } } },
  });
}

const _ACTION_COLOR = {
  "On Phone":  "#f44336",
  "Sleeping":  "#9c27b0",
  "Writing":   "#2196f3",
  "Talking":   "#ff9800",
  "Attentive": "#4caf50",
  "Distracted":"#607d8b",
};

function renderCombinedStudents(students) {
  if (!students.length) {
    combStudents.innerHTML = '<p style="color:#888;padding:12px">No students tracked.</p>';
    return;
  }

  const rows = students.map((s, idx) => {
    const attnClass  = s.attention_status === "High" ? "attn-high" : s.attention_status === "Medium" ? "attn-medium" : "attn-low";
    const attendClass = s.attendance === "Present" ? "attend-present" : "attend-absent";
    const concPct    = Number(s.concentration_pct || 0);
    const concBar    = `<div class="conc-bar-bg"><div class="conc-bar-fill" style="width:${Math.min(100, concPct)}%"></div></div>`;
    const actColor   = _ACTION_COLOR[s.detected_action] || "#607d8b";
    const clipCell   = s.clip_url
      ? `<video class="cog-clip-thumb" controls preload="metadata" src="${s.clip_url}"></video>`
      : `<span style="color:#aaa">—</span>`;

    const bodyPcts = [
      s.writing_pct  != null ? `W ${engPct(s.writing_pct)}`  : null,
      s.sleeping_pct != null ? `Z ${engPct(s.sleeping_pct)}` : null,
      s.phone_pct    != null ? `📱 ${engPct(s.phone_pct)}`    : null,
    ].filter(Boolean).join("  ");

    return `
      <tr>
        <td>${idx + 1}</td>
        <td>${s.student_label || `Student ${s.track_id}`}</td>
        <td><span style="color:${actColor};font-weight:600">${s.detected_action || "—"}</span></td>
        <td><span class="attn-badge ${attnClass}">${s.attention_status}</span></td>
        <td>${(s.emotion || "—").charAt(0).toUpperCase() + (s.emotion || "").slice(1)}</td>
        <td>${concPct.toFixed(1)}% ${concBar}</td>
        <td>${s.engagement_score ?? "—"}</td>
        <td style="font-size:0.78rem;white-space:nowrap">${bodyPcts || "—"}</td>
        <td>${s.detected_time_s ?? "—"}s</td>
        <td class="${attendClass}">${s.attendance}</td>
        <td>${s.blink_count ?? "—"}</td>
        <td>${clipCell}</td>
      </tr>`;
  }).join("");

  combStudents.innerHTML = `
    <div class="cognitive-table-wrap">
      <table class="cognitive-table">
        <thead>
          <tr>
            <th>#</th>
            <th>Student</th>
            <th>Action</th>
            <th>Attention</th>
            <th>Emotion</th>
            <th>Concentration</th>
            <th>Engage</th>
            <th>Body breakdown</th>
            <th>Time seen</th>
            <th>Attendance</th>
            <th>Blinks</th>
            <th>Clip</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;
}

// ── Classroom tab ─────────────────────────────────────────────────────────────

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

classroomFileInput.addEventListener("change", () => {
  classroomFileLabel.textContent =
    selectedFileText(classroomFileInput.files, "Choose a classroom video (up to 60 min)");
});

const ACTION_COLORS = {
  "Attentive":  "#22c55e",
  "Writing":    "#3b82f6",
  "Talking":    "#f59e0b",
  "On Phone":   "#ef4444",
  "Sleeping":   "#8b5cf6",
  "Distracted": "#f97316",
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
  classroomStatus.textContent = "Analysing classroom… This can take several minutes for long videos.";
  btn.disabled = true;

  try {
    const resp = await fetch("/api/classroom/process", { method: "POST", body: payload });
    const data = await resp.json();
    if (!data.ok) throw new Error(data.error || "Unknown error");

    const s = data.summary;
    classroomStatus.textContent =
      `Done — ${s.student_count} students detected across ${s.total_windows} windows ` +
      `(${s.duration_seconds}s video, processed in ${s.processing_time_s}s).`;

    classroomMetrics.innerHTML = `
      <div class="metric"><span class="metric-value">${s.student_count}</span><span class="metric-label">Students</span></div>
      <div class="metric"><span class="metric-value">${s.total_windows}</span><span class="metric-label">Windows</span></div>
      <div class="metric"><span class="metric-value">${s.class_attentive_pct}%</span><span class="metric-label">Class attentive</span></div>
      <div class="metric"><span class="metric-value">${Math.round(s.duration_seconds / 60)}m</span><span class="metric-label">Duration</span></div>`;

    classroomSummaryLink.href = data.download_urls.summary_json;
    classroomCsvLink.href     = data.download_urls.csv;

    const actionTotals = {};
    let totalObs = 0;
    for (const student of (s.students || [])) {
      for (const win of (student.timeline || [])) {
        actionTotals[win.action] = (actionTotals[win.action] || 0) + 1;
        totalObs++;
      }
    }
    const barSegs = Object.entries(actionTotals)
      .sort((a, b) => b[1] - a[1])
      .map(([action, count]) => {
        const pct = totalObs ? (count / totalObs * 100).toFixed(1) : 0;
        return `<div class="cls-bar-seg" style="flex:${count};background:${actionColor(action)}" title="${action}: ${pct}%">
                  <span>${action} ${pct}%</span>
                </div>`;
      }).join("");
    classroomClassBar.innerHTML =
      `<p class="cls-bar-label">Class-wide action distribution</p><div class="cls-bar">${barSegs}</div>`;

    renderClassroomStudents(s.students || []);
    classroomResults.classList.remove("hidden");
  } catch (err) {
    classroomStatus.textContent = `Error: ${err.message}`;
    classroomStatus.classList.add("error");
  } finally {
    btn.disabled = false;
  }
});

function renderClassroomStudents(students) {
  if (!students.length) {
    classroomStudents.innerHTML = "<p class='empty-state'>No students detected.</p>";
    return;
  }

  classroomStudents.innerHTML = students.map((student) => {
    const breakdownBars = Object.entries(student.action_breakdown || {})
      .sort((a, b) => b[1] - a[1])
      .map(([action, pct]) =>
        `<div class="cls-mini-seg" style="flex:${pct};background:${actionColor(action)}" title="${action}: ${pct}%"></div>`
      ).join("");

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
          : `<div class="cls-win-no-clip">no clip</div>`
        }
      </div>`;
    }).join("");

    const attColor = student.attentive_pct >= 70 ? "#22c55e"
                   : student.attentive_pct >= 40 ? "#f59e0b" : "#ef4444";

    return `<details class="cls-student-card" open>
      <summary class="cls-student-summary">
        <span class="cls-student-id">${student.student_label}</span>
        <span class="cls-student-dominant">${student.dominant_action}</span>
        <span class="cls-student-attn" style="color:${attColor}">${student.attentive_pct}% attentive</span>
        <span class="cls-student-windows">${student.windows_seen} windows</span>
        <div class="cls-mini-bar">${breakdownBars}</div>
      </summary>
      <div class="cls-timeline">${timelineHtml}</div>
    </details>`;
  }).join("");
}
