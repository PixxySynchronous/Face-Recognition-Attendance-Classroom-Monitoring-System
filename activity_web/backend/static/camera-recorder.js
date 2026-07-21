// Shared camera-recording widget, used by both the teacher dashboard's enroll
// tab and the student self-enrollment page. Records a short webm clip via
// MediaRecorder and hands the Blob back to the caller — it never touches a
// file <input>, the caller just asks for the current Blob on submit.
//
// Walks the person through a short guided sequence (look center, turn left,
// turn right, ...) while recording, and overlays a face-position guide, so
// the resulting clip has the pose variety the enrollment pipeline wants
// (recall ANCHOR_CONSISTENCY_THRESHOLD tolerates moderate turns, and the
// degraded/distance copies it generates work better from a clear, centered,
// well-lit face). The sequence is just a default — pass `script` to override.
//
// Usage:
//   const recorder = CameraRecorder.create(containerEl);
//   ...
//   const blob = recorder.getBlob(); // null if nothing recorded yet
//   recorder.reset();                // clear after a successful submit

const CameraRecorder = (() => {
  const DEFAULT_SCRIPT = [
    { text: "Look straight at the camera", seconds: 10 },
    { text: "Slowly turn your head to the left", seconds: 8 },
    { text: "Slowly turn your head to the right", seconds: 8 },
    { text: "Back to center. Almost done", seconds: 6 },
  ];

  function speak(text) {
    if (!("speechSynthesis" in window)) return;
    try {
      window.speechSynthesis.cancel(); // don't queue/overlap with a prior step
      const utter = new SpeechSynthesisUtterance(text);
      utter.rate = 0.95;
      window.speechSynthesis.speak(utter);
    } catch (err) {
      console.warn("Speech synthesis unavailable:", err);
    }
  }

  function create(container, options = {}) {
    const script = options.script && options.script.length ? options.script : DEFAULT_SCRIPT;

    container.innerHTML = `
      <div class="camera-recorder">
        <div class="camera-video-wrap">
          <video class="camera-preview" autoplay muted playsinline></video>
          <div class="camera-face-guide" hidden></div>
          <div class="camera-instruction" hidden></div>
        </div>
        <p class="camera-hint">Center your face in the oval, in good light, at arm's length from the camera.</p>
        <div class="camera-controls">
          <button type="button" class="camera-btn" data-action="start-camera">Start camera</button>
          <button type="button" class="camera-btn" data-action="record" disabled>Start recording</button>
          <button type="button" class="camera-btn" data-action="retake" hidden>Retake</button>
        </div>
        <div class="camera-status">Camera off.</div>
      </div>
    `;

    const videoEl       = container.querySelector(".camera-preview");
    const faceGuideEl   = container.querySelector(".camera-face-guide");
    const instructionEl = container.querySelector(".camera-instruction");
    const statusEl      = container.querySelector(".camera-status");
    const startBtn       = container.querySelector('[data-action="start-camera"]');
    const recordBtn      = container.querySelector('[data-action="record"]');
    const retakeBtn      = container.querySelector('[data-action="retake"]');

    let stream = null;
    let mediaRecorder = null;
    let chunks = [];
    let recordedBlob = null;
    let recording = false;
    let scriptTimer = null;

    async function startCamera() {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      } catch (err) {
        statusEl.textContent = "Couldn't access the camera: " + err.message;
        return;
      }
      videoEl.srcObject = stream;
      faceGuideEl.hidden = false;
      startBtn.disabled = true;
      recordBtn.disabled = false;
      statusEl.textContent = "Camera on. Center your face in the oval, then start recording.";
    }

    function runGuidedScript() {
      let stepIndex = 0;
      const showStep = () => {
        if (!recording || stepIndex >= script.length) {
          instructionEl.hidden = true;
          if (recording) stopRecording();
          return;
        }
        const step = script[stepIndex];
        instructionEl.hidden = false;
        instructionEl.textContent = `${step.text} (${stepIndex + 1}/${script.length})`;
        speak(step.text);
        stepIndex++;
        scriptTimer = setTimeout(showStep, step.seconds * 1000);
      };
      showStep();
    }

    function startRecording() {
      if (!stream) return;
      chunks = [];
      const mimeType = MediaRecorder.isTypeSupported("video/webm;codecs=vp8")
        ? "video/webm;codecs=vp8"
        : "video/webm";
      mediaRecorder = new MediaRecorder(stream, { mimeType });
      mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) chunks.push(e.data); };
      mediaRecorder.onstop = () => {
        recordedBlob = new Blob(chunks, { type: mimeType });
        videoEl.srcObject = null;
        videoEl.src = URL.createObjectURL(recordedBlob);
        videoEl.muted = false;
        videoEl.controls = true;
        videoEl.play().catch(() => {});
        stream.getTracks().forEach((t) => t.stop());
        stream = null;
        recording = false;
        faceGuideEl.hidden = true;
        instructionEl.hidden = true;
        recordBtn.hidden = true;
        retakeBtn.hidden = false;
        statusEl.textContent = "Recorded. Review it, or retake.";
      };
      mediaRecorder.start();
      recording = true;
      recordBtn.textContent = "Stop recording";
      statusEl.textContent = "Recording…";
      runGuidedScript();
    }

    function stopRecording() {
      if (scriptTimer) { clearTimeout(scriptTimer); scriptTimer = null; }
      if ("speechSynthesis" in window) window.speechSynthesis.cancel();
      if (mediaRecorder && recording) mediaRecorder.stop();
    }

    function retake() {
      if (scriptTimer) { clearTimeout(scriptTimer); scriptTimer = null; }
      if ("speechSynthesis" in window) window.speechSynthesis.cancel();
      recordedBlob = null;
      videoEl.controls = false;
      videoEl.muted = true;
      videoEl.removeAttribute("src");
      videoEl.load();
      faceGuideEl.hidden = true;
      instructionEl.hidden = true;
      recordBtn.hidden = false;
      recordBtn.textContent = "Start recording";
      retakeBtn.hidden = true;
      startBtn.disabled = false;
      recordBtn.disabled = true;
      statusEl.textContent = "Camera off.";
    }

    startBtn.addEventListener("click", startCamera);
    recordBtn.addEventListener("click", () => (recording ? stopRecording() : startRecording()));
    retakeBtn.addEventListener("click", retake);

    return {
      getBlob: () => recordedBlob,
      reset: retake,
      stopStream: () => {
        if (scriptTimer) { clearTimeout(scriptTimer); scriptTimer = null; }
        if ("speechSynthesis" in window) window.speechSynthesis.cancel();
        if (stream) stream.getTracks().forEach((t) => t.stop());
      },
    };
  }

  return { create, DEFAULT_SCRIPT };
})();
