/* dashboard.js — main application logic */
/* Initial values are read from the DOM to avoid Jinja coupling */

/* ── Element refs ── */
const conf               = document.getElementById('conf');
const confText           = document.getElementById('confText');
const counts             = document.getElementById('counts');
const perf               = document.getElementById('perf');
const analyticsFps       = document.getElementById('analyticsFps');
const analyticsInference = document.getElementById('analyticsInference');
const analyticsObjects   = document.getElementById('analyticsObjects');
const analyticsConfidence= document.getElementById('analyticsConfidence');
const activeCamera       = document.getElementById('activeCamera');
const feedGrid           = document.getElementById('feedGrid');
const inferenceToggle    = document.getElementById('inferenceToggle');
const inferenceStatus    = document.getElementById('inferenceStatus');
const cameraSelect       = document.getElementById('cameraSelect');
const cameraCountBadge   = document.getElementById('cameraCountBadge');
const cameraLabel        = document.getElementById('cameraLabel');
const cameraRef          = document.getElementById('cameraRef');
const cameraStatus       = document.getElementById('cameraStatus');
const availabilityStatus = document.getElementById('availabilityStatus');
const addCameraStatus    = document.getElementById('addCameraStatus');
const cameraList         = document.getElementById('cameraList');
const testCamera         = document.getElementById('testCamera');
const addCamera          = document.getElementById('addCamera');
const removeCamera       = document.getElementById('removeCamera');
const startCamera        = document.getElementById('startCamera');
const stopCamera         = document.getElementById('stopCamera');
const uploadInput        = document.getElementById('uploadInput');
const uploadBtn          = document.getElementById('uploadBtn');
const uploadStatus       = document.getElementById('uploadStatus');

/* ── State (seeded from DOM) ── */
let cameras          = [];
let activeRef        = (activeCamera && activeCamera.value) || '';
let maxCameras       = 5;
let inferenceEnabled = false;

/* Seed conf display from slider's initial value */
if (conf && confText) confText.textContent = Number(conf.value).toFixed(2);

/* ── Helpers ── */
function setStatus(el, text, cls) {
  if (!el) return;
  el.textContent = text;
  el.classList.remove('ok', 'warn', 'bad');
  el.classList.add(cls);
}

/* ── Inference state ── */
function renderInferenceState() {
  inferenceToggle.classList.remove('btn-green', 'btn-red');
  if (inferenceEnabled) {
    inferenceToggle.textContent = 'Stop Inference';
    inferenceToggle.classList.add('btn-red');
    setStatus(inferenceStatus, 'Inference running', 'ok');
  } else {
    inferenceToggle.textContent = 'Start Inference';
    inferenceToggle.classList.add('btn-green');
    setStatus(inferenceStatus, 'Inference paused', 'warn');
  }
}

/* ── Render cameras dropdown + tab list ── */
function renderCameras() {
  /* Dropdown in Live Feeds */
  cameraSelect.innerHTML = '';
  if (cameras.length === 0) {
    const opt = document.createElement('option');
    opt.textContent = 'No cameras registered';
    opt.disabled = true;
    opt.selected = true;
    cameraSelect.appendChild(opt);
  } else {
    cameras.forEach((camera) => {
      const opt = document.createElement('option');
      opt.value = String(camera.id);
      opt.textContent = camera.id === activeRef
        ? camera.label + ' (' + camera.ref + ') \u2014 active'
        : camera.label + ' (' + camera.ref + ')';
      if (camera.id === activeRef) opt.selected = true;
      cameraSelect.appendChild(opt);
    });
  }
  cameraCountBadge.textContent = cameras.length + ' / ' + maxCameras;
  cameraCountBadge.className = 'badge ' + (cameras.length >= maxCameras ? 'badge-red' : 'badge-amber');

  /* List in Cameras tab */
  if (!cameraList) return;
  if (cameras.length === 0) {
    cameraList.textContent = 'No cameras registered.';
  } else {
    cameraList.innerHTML = cameras.map((c) => {
      const running = c.running === true;
      return '<div style="display:flex;align-items:center;justify-content:space-between;padding:6px 0;border-bottom:1px solid var(--border);">' +
        '<span><strong>' + c.label + '</strong> <span style="color:var(--muted);font-size:12px;">(' + c.ref + ')</span></span>' +
        '<span id="listRunning-' + c.id + '" class="status ' + (running ? 'ok' : 'warn') + '" style="font-size:11px;">' +
          (running ? 'Running' : 'Stopped') +
        '</span>' +
        '</div>';
    }).join('');
  }
}

/* ── Render feed grid ── */
function renderFeedGrid() {
  feedGrid.innerHTML = '';
  if (!cameras.length) {
    const empty = document.createElement('div');
    empty.className = 'counts-box';
    empty.style.gridColumn = '1/-1';
    empty.innerHTML = 'No cameras added yet. Go to the <strong>Cameras</strong> tab to register one.';
    feedGrid.appendChild(empty);
    return;
  }
  cameras.forEach((camera) => {
    const isRunning = camera.running === true;
    const card = document.createElement('div');
    card.className = 'feed-card';
    card.id = 'feedCard-' + camera.id;
    card.innerHTML =
      '<div class="feed-card-header">' +
        '<strong>' + camera.label + '</strong>' +
        '<span id="camRunning-' + camera.id + '" class="status ' + (isRunning ? 'ok' : 'warn') + '">' +
          (isRunning ? 'Running' : 'Stopped') +
        '</span>' +
      '</div>' +
      '<img src="/video_feed/' + camera.id + '" alt="' + camera.label + '" />' +
      '<div class="feed-card-footer">' +
        '<span id="camFps-' + camera.id + '">FPS: 0</span>' +
        '<span id="camInf-' + camera.id + '">INF: 0 ms</span>' +
      '</div>';
    feedGrid.appendChild(card);
  });
}

/* ── API ── */
async function loadCameras() {
  try {
    const res  = await fetch('/cameras');
    const data = await res.json();
    cameras    = Array.isArray(data.cameras) ? data.cameras : [];
    activeRef  = data.active || activeRef;
    maxCameras = Number(data.max) || maxCameras;
    renderCameras();
    renderFeedGrid();
  } catch { /* silent */ }
}

async function setActiveCamera(id) {
  try {
    const res  = await fetch('/cameras/active', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id }),
    });
    const data = await res.json();
    activeRef = data.active || id;
    renderCameras();
    if (activeCamera) activeCamera.value = activeRef;
    setStatus(cameraStatus, 'Active camera updated', 'ok');
  } catch { setStatus(cameraStatus, 'Failed to update active camera', 'bad'); }
}

async function setInferenceState(enabled) {
  try {
    const res  = await fetch('/inference', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled }),
    });
    if (!res.ok) throw new Error();
    const data = await res.json();
    inferenceEnabled = Boolean(data.inference_enabled);
    renderInferenceState();
  } catch { setStatus(inferenceStatus, 'Toggle failed', 'bad'); }
}

async function updateConfig(value) {
  try {
    await fetch('/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ confidence: Number(value) }),
    });
  } catch { /* silent */ }
}

/* ── Events ── */
conf.addEventListener('input', (e) => {
  const v = Number(e.target.value).toFixed(2);
  confText.textContent = v;
  if (analyticsConfidence) analyticsConfidence.textContent = v;
});
conf.addEventListener('change', (e) => updateConfig(e.target.value));

inferenceToggle.addEventListener('click', () => setInferenceState(!inferenceEnabled));

cameraSelect.addEventListener('change', () => {
  const id = cameraSelect.value;
  if (id) setActiveCamera(id);
});

startCamera.addEventListener('click', async () => {
  const id = cameraSelect.value || activeRef;
  if (!id) { setStatus(cameraStatus, 'Select a camera first', 'warn'); return; }
  setStatus(availabilityStatus, 'Starting...', 'warn');
  try {
    const res = await fetch('/cameras/' + id + '/start', { method: 'POST' });
    if (!res.ok) throw new Error();
    setStatus(availabilityStatus, 'Camera started', 'ok');
    const el = document.getElementById('camRunning-' + id);
    if (el) { el.textContent = 'Running'; el.className = 'status ok'; }
  } catch { setStatus(availabilityStatus, 'Failed to start', 'bad'); }
});

stopCamera.addEventListener('click', async () => {
  const id = cameraSelect.value || activeRef;
  if (!id) { setStatus(cameraStatus, 'Select a camera first', 'warn'); return; }
  setStatus(availabilityStatus, 'Stopping...', 'warn');
  try {
    const res = await fetch('/cameras/' + id + '/stop', { method: 'POST' });
    if (!res.ok) throw new Error();
    setStatus(availabilityStatus, 'Camera stopped', 'warn');
    const el = document.getElementById('camRunning-' + id);
    if (el) { el.textContent = 'Stopped'; el.className = 'status warn'; }
  } catch { setStatus(availabilityStatus, 'Failed to stop', 'bad'); }
});

testCamera.addEventListener('click', async () => {
  const ref = cameraRef.value.trim();
  const id  = cameraSelect.value || activeRef;
  setStatus(addCameraStatus, 'Testing...', 'warn');
  try {
    const res  = await fetch('/cameras/test', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(ref ? { ref } : { id }),
    });
    const data = await res.json();
    setStatus(addCameraStatus, data.status === 'ok' ? 'Stream is live' : 'Stream error',
                               data.status === 'ok' ? 'ok' : 'bad');
  } catch { setStatus(addCameraStatus, 'Stream error', 'bad'); }
});

addCamera.addEventListener('click', async () => {
  const label = cameraLabel.value.trim();
  const ref   = cameraRef.value.trim();
  if (!label || !ref) { setStatus(addCameraStatus, 'Label and reference required', 'warn'); return; }
  if (cameras.length >= maxCameras) { setStatus(addCameraStatus, 'Max ' + maxCameras + ' cameras', 'warn'); return; }
  try {
    const res = await fetch('/cameras', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ label, ref }),
    });
    if (!res.ok) { const e = await res.json(); throw new Error(e.error || 'error'); }
    const data = await res.json();
    cameras    = Array.isArray(data.cameras) ? data.cameras : cameras;
    maxCameras = Number(data.max) || maxCameras;
    cameraLabel.value = '';
    cameraRef.value   = '';
    renderCameras();
    renderFeedGrid();
    setStatus(addCameraStatus, 'Camera added', 'ok');
  } catch (err) { setStatus(addCameraStatus, err.message || 'Failed to add camera', 'bad'); }
});

removeCamera.addEventListener('click', async () => {
  const id = cameraSelect.value;
  if (!id) { setStatus(addCameraStatus, 'Select a camera from the dropdown first', 'warn'); return; }
  try {
    const res  = await fetch('/cameras/' + id, { method: 'DELETE' });
    const data = await res.json();
    cameras = Array.isArray(data.cameras) ? data.cameras : [];
    renderCameras();
    renderFeedGrid();
    setStatus(addCameraStatus, 'Camera removed', 'ok');
  } catch { setStatus(addCameraStatus, 'Failed to remove camera', 'bad'); }
});

uploadInput.addEventListener('change', () => {
  const file = uploadInput.files && uploadInput.files[0];
  setStatus(uploadStatus, file ? 'Ready: ' + file.name : 'No file selected', file ? 'ok' : 'warn');
});

uploadBtn.addEventListener('click', async () => {
  const file = uploadInput.files && uploadInput.files[0];
  if (!file) { setStatus(uploadStatus, 'Select a file first', 'warn'); return; }
  const form = new FormData();
  form.append('file', file);
  try {
    const res  = await fetch('/upload', { method: 'POST', body: form });
    const data = await res.json();
    setStatus(uploadStatus,
      (res.ok && data.status === 'ok') ? 'Upload complete' : 'Upload failed',
      (res.ok && data.status === 'ok') ? 'ok' : 'bad');
  } catch { setStatus(uploadStatus, 'Upload failed', 'bad'); }
});

/* ── Stats polling ── */
function parseTotal(text) {
  if (!text || text.includes('No objects')) return 0;
  return text.split('\n')
    .map((l) => l.split(':'))
    .filter((p) => p.length === 2)
    .map((p) => Number(p[1].trim()))
    .filter((v) => !isNaN(v))
    .reduce((s, v) => s + v, 0);
}

async function refreshStats() {
  try {
    const res    = await fetch('/stats');
    const data   = await res.json();
    const totals = data.totals || {};
    const ct     = totals.counts || 'No objects detected.';

    if (counts)             counts.textContent             = ct;
    if (perf)               perf.textContent               = 'FPS: ' + (totals.fps ?? 0) + '\nInference: ' + (totals.inference_ms ?? 0) + ' ms';
    if (analyticsFps)       analyticsFps.textContent       = totals.fps ?? 0;
    if (analyticsInference) analyticsInference.textContent = totals.inference_ms ?? 0;
    if (analyticsObjects)   analyticsObjects.textContent   = totals.objects ?? parseTotal(ct);
    if (analyticsConfidence)analyticsConfidence.textContent= Number(data.confidence ?? 0).toFixed(2);

    if (typeof data.inference_enabled === 'boolean') {
      inferenceEnabled = data.inference_enabled;
      renderInferenceState();
    }

    if (data.active_camera_id) {
      activeRef = data.active_camera_id;
      if (activeCamera) activeCamera.value = data.active_camera_id;
    }

    (data.cameras || []).forEach((cam) => {
      const fps     = document.getElementById('camFps-' + cam.id);
      const inf     = document.getElementById('camInf-' + cam.id);
      const run     = document.getElementById('camRunning-' + cam.id);
      const listRun = document.getElementById('listRunning-' + cam.id);

      if (fps) fps.textContent = 'FPS: ' + (cam.fps ?? 0).toFixed(1);
      if (inf) inf.textContent = 'INF: ' + (cam.inference_ms ?? 0).toFixed(1) + ' ms';

      const running = cam.running === true;
      const label   = running ? 'Running' : 'Stopped';
      const cls     = 'status ' + (running ? 'ok' : 'warn');

      if (run)     { run.textContent     = label; run.className     = cls; }
      if (listRun) { listRun.textContent = label; listRun.className = cls; }
    });
  } catch { /* silent */ }
}

/* ── Boot ── */
loadCameras();
renderInferenceState();
setInterval(refreshStats, 500);
refreshStats();
