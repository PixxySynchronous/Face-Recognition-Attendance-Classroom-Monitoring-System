// Student self-enrollment page. Posts to the same /api/attendance/enroll
// endpoint the teacher dashboard uses — just a lighter-weight form around it.

const classroomSelect = document.getElementById("self-enroll-classroom");
const nameInput       = document.getElementById("self-enroll-name");
const mediaInput      = document.getElementById("self-enroll-media-input");
const mediaLabel      = document.getElementById("self-enroll-media-label");
const form            = document.getElementById("self-enroll-form");
const statusEl        = document.getElementById("self-enroll-status");
const resultEl        = document.getElementById("self-enroll-result");

let enrollTab = "files";
let cameraRecorder = null;

function selectedFileText(files, fallback) {
  if (!files || !files.length) return fallback;
  return files.length === 1 ? files[0].name : `${files[0].name} + ${files.length - 1} more`;
}

mediaInput.addEventListener("change", () => {
  mediaLabel.textContent = selectedFileText(mediaInput.files, "Choose photos or a short video");
});

function switchSelfEnrollTab(tab) {
  const previousTab = enrollTab;
  enrollTab = tab;
  document.getElementById("self-enroll-tab-files").style.display  = tab === "files"  ? "" : "none";
  document.getElementById("self-enroll-tab-camera").style.display = tab === "camera" ? "" : "none";
  document.getElementById("self-tab-files").classList.toggle("tab-active",  tab === "files");
  document.getElementById("self-tab-camera").classList.toggle("tab-active", tab === "camera");

  if (tab === "camera" && !cameraRecorder) {
    cameraRecorder = CameraRecorder.create(document.getElementById("self-enroll-camera-recorder"));
  }
  if (previousTab === "camera" && tab !== "camera" && cameraRecorder) {
    cameraRecorder.stopStream();
  }
}

async function loadClassrooms() {
  try {
    const response = await fetch("/api/attendance/classrooms");
    const data = await response.json();
    if (!data.ok || !data.classrooms.length) throw new Error(data.error || "No classrooms available.");
    classroomSelect.innerHTML = data.classrooms.map((c) => `<option value="${c.id}">${c.label}</option>`).join("");
  } catch (err) {
    statusEl.textContent = "Couldn't load classrooms: " + err.message;
    statusEl.classList.add("error");
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const name = nameInput.value.trim();
  if (!name) { statusEl.textContent = "Enter your name."; statusEl.classList.add("error"); return; }

  const btn = form.querySelector("button[type='submit']");
  statusEl.classList.remove("error");
  resultEl.classList.add("hidden");
  btn.disabled = true;

  try {
    const payload = new FormData();
    payload.append("classroom", classroomSelect.value);
    payload.append("student_name", name);

    if (enrollTab === "camera") {
      const blob = cameraRecorder && cameraRecorder.getBlob();
      if (!blob) { statusEl.textContent = "Record a video first."; statusEl.classList.add("error"); btn.disabled = false; return; }
      payload.append("media", blob, "recording.webm");
    } else {
      if (!mediaInput.files.length) { statusEl.textContent = "Upload at least one photo or video."; statusEl.classList.add("error"); btn.disabled = false; return; }
      Array.from(mediaInput.files).forEach((f) => payload.append("media", f));
    }

    statusEl.textContent = "Extracting your face profile...";
    const response = await fetch("/api/attendance/enroll", { method: "POST", body: payload });
    const data = await response.json();
    if (!response.ok || !data.ok) throw new Error(data.error || "Enrollment failed.");

    statusEl.textContent = `You're enrolled, ${data.student.name}!`;
    resultEl.classList.remove("hidden");
    resultEl.innerHTML = `<div class="result-summary">${data.student.name} — ${data.student.observations ?? 0} embeddings captured</div>`;
    if (cameraRecorder) cameraRecorder.reset();
    form.reset();
    mediaLabel.textContent = "Choose photos or a short video";
  } catch (err) {
    statusEl.textContent = err.message;
    statusEl.classList.add("error");
  } finally {
    btn.disabled = false;
  }
});

loadClassrooms();
