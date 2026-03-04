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
const startCamera        = document.getElementById('startCamera');
const stopCamera         = document.getElementById('stopCamera');
const removeCamera       = document.getElementById('removeCamera'); // may be null
const uploadInput        = document.getElementById('uploadInput');
const uploadBtn          = document.getElementById('uploadBtn');
const uploadStatus       = document.getElementById('uploadStatus');
const uploadInferenceStatus  = document.getElementById('uploadInferenceStatus');
const uploadInferenceSummary = document.getElementById('uploadInferenceSummary');
const uploadInferencePreview = document.getElementById('uploadInferencePreview');
const uploadStreamCanvas     = document.getElementById('uploadStreamCanvas');
const uploadReplayBtn         = document.getElementById('uploadReplayBtn');

let uploadStream = null;
let uploadFrames = [];
let replayTimer = null;
const uploadStreamFps = 15;

function stopReplay() {
  if (replayTimer) {
    clearInterval(replayTimer);
    replayTimer = null;
  }
}

function startReplay() {
  if (!uploadStreamCanvas || uploadFrames.length === 0) return;
  stopReplay();
  let idx = 0;
  const ctx = uploadStreamCanvas.getContext('2d');
  replayTimer = setInterval(() => {
    const frame = uploadFrames[idx];
    if (!frame) return;
    const img = new Image();
    img.onload = () => {
      uploadStreamCanvas.width = img.naturalWidth;
      uploadStreamCanvas.height = img.naturalHeight;
      ctx.drawImage(img, 0, 0);
    };
    img.src = frame;
    idx += 1;
    if (idx >= uploadFrames.length) {
      stopReplay();
    }
  }, Math.round(1000 / uploadStreamFps));
}

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
    cameraList.innerHTML = '<div class="counts-box">No cameras registered.</div>';
  } else {
    cameraList.innerHTML = cameras.map((c) => {
      const running = c.running === true;
      return '<div style="display:flex;align-items:center;gap:8px;padding:8px 0;border-bottom:1px solid var(--border);flex-wrap:wrap;">' +
        '<div style="flex:1;min-width:100px;">' +
          '<strong>' + c.label + '</strong>' +
          '<span style="color:var(--muted);font-size:11px;margin-left:6px;">ref: ' + c.ref + '</span>' +
        '</div>' +
        '<span id="listRunning-' + c.id + '" class="status ' + (running ? 'ok' : 'warn') + '" style="font-size:11px;">' +
          (running ? 'Running' : 'Stopped') +
        '</span>' +
        '<button class="btn btn-green" style="padding:2px 10px;font-size:12px;" data-action="start" data-id="' + c.id + '">Start</button>' +
        '<button class="btn btn-red"   style="padding:2px 10px;font-size:12px;" data-action="stop"  data-id="' + c.id + '">Stop</button>' +
        '<button class="btn btn-ghost" style="padding:2px 10px;font-size:12px;" data-action="delete" data-id="' + c.id + '">&#x1F5D1;</button>' +
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
      '</div>' +
      '<div id="camError-' + camera.id + '" class="feed-error" style="display: none; padding: 10px 14px; background: var(--danger-bg); border-top: 1px solid var(--danger-border); color: var(--danger); font-size: 11px; font-family: monospace;"></div>';
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
    if (!res.ok) {
      let errMsg = 'error';
      try { const e = await res.json(); errMsg = e.error || 'error'; } catch { errMsg = `Server error (${res.status})`; }
      throw new Error(errMsg);
    }
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

removeCamera && removeCamera.addEventListener('click', async () => {
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

/* ── Camera list event delegation (start / stop / delete per row) ── */
cameraList && cameraList.addEventListener('click', async (e) => {
  const btn = e.target.closest('[data-action]');
  if (!btn) return;
  const action = btn.dataset.action;
  const id     = btn.dataset.id;
  if (!id) return;

  if (action === 'start') {
    setStatus(addCameraStatus, 'Starting…', 'warn');
    try {
      const res = await fetch('/cameras/' + id + '/start', { method: 'POST' });
      if (!res.ok) throw new Error();
      setStatus(addCameraStatus, 'Camera started', 'ok');
      const el = document.getElementById('listRunning-' + id);
      if (el) { el.textContent = 'Running'; el.className = 'status ok'; }
      const el2 = document.getElementById('camRunning-' + id);
      if (el2) { el2.textContent = 'Running'; el2.className = 'status ok'; }
    } catch { setStatus(addCameraStatus, 'Failed to start', 'bad'); }

  } else if (action === 'stop') {
    setStatus(addCameraStatus, 'Stopping…', 'warn');
    try {
      const res = await fetch('/cameras/' + id + '/stop', { method: 'POST' });
      if (!res.ok) throw new Error();
      setStatus(addCameraStatus, 'Camera stopped', 'warn');
      const el = document.getElementById('listRunning-' + id);
      if (el) { el.textContent = 'Stopped'; el.className = 'status warn'; }
      const el2 = document.getElementById('camRunning-' + id);
      if (el2) { el2.textContent = 'Stopped'; el2.className = 'status warn'; }
    } catch { setStatus(addCameraStatus, 'Failed to stop', 'bad'); }

  } else if (action === 'delete') {
    if (!confirm('Delete camera ' + id + '?')) return;
    try {
      const res  = await fetch('/cameras/' + id, { method: 'DELETE' });
      const data = await res.json();
      cameras = Array.isArray(data.cameras) ? data.cameras : [];
      renderCameras();
      renderFeedGrid();
      setStatus(addCameraStatus, 'Camera removed', 'ok');
    } catch { setStatus(addCameraStatus, 'Failed to remove', 'bad'); }
  }
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

    if (res.ok && data.status === 'ok' && data.path) {
      setStatus(uploadInferenceStatus, 'Processing…', 'warn');
      if (uploadInferenceSummary) uploadInferenceSummary.textContent = 'Running inference...';
      if (uploadInferencePreview) uploadInferencePreview.style.display = 'none';
      if (uploadStreamCanvas) uploadStreamCanvas.style.display = 'none';
      if (uploadReplayBtn) uploadReplayBtn.disabled = true;
      uploadFrames = [];
      stopReplay();

      if (uploadStream) {
        uploadStream.close();
        uploadStream = null;
      }

      const streamUrl = '/upload/stream?path=' + encodeURIComponent(data.path) + '&fps=15&seconds=0';
      uploadStream = new EventSource(streamUrl);

      uploadStream.addEventListener('frame', (evt) => {
        const payload = JSON.parse(evt.data || '{}');
        if (uploadInferenceStatus) setStatus(uploadInferenceStatus, 'Streaming…', 'ok');
        if (uploadInferenceSummary) {
          const lines = [
            'Frame: ' + (payload.frame || 0),
            'Inference: ' + Number(payload.inference_ms || 0).toFixed(1) + ' ms',
            'Counts:',
            payload.counts_text || 'No objects detected.',
          ];
          uploadInferenceSummary.textContent = lines.join('\n');
        }

        if (uploadStreamCanvas && payload.jpeg) {
          const dataUrl = 'data:image/jpeg;base64,' + payload.jpeg;
          uploadFrames.push(dataUrl);
          const img = new Image();
          img.onload = () => {
            uploadStreamCanvas.width = img.naturalWidth;
            uploadStreamCanvas.height = img.naturalHeight;
            uploadStreamCanvas.getContext('2d').drawImage(img, 0, 0);
            uploadStreamCanvas.style.display = 'block';
          };
          img.src = dataUrl;
        }
      });

      uploadStream.addEventListener('done', () => {
        if (uploadInferenceStatus) setStatus(uploadInferenceStatus, 'Inference complete', 'ok');
        if (uploadStream) {
          uploadStream.close();
          uploadStream = null;
        }
        if (uploadReplayBtn) uploadReplayBtn.disabled = uploadFrames.length === 0;
      });

      uploadStream.addEventListener('error', () => {
        if (uploadInferenceStatus) setStatus(uploadInferenceStatus, 'Stream error', 'bad');
        if (uploadStream) {
          uploadStream.close();
          uploadStream = null;
        }
        if (uploadReplayBtn) uploadReplayBtn.disabled = uploadFrames.length === 0;
      });
    }
  } catch { setStatus(uploadStatus, 'Upload failed', 'bad'); }
});

uploadReplayBtn && uploadReplayBtn.addEventListener('click', () => {
  startReplay();
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
    const analytics = data.analytics || {};

    const combinedCountsText = analytics.combined_counts_text || ct;
    if (counts)             counts.textContent             = combinedCountsText;
    if (perf) {
      const uploadsInf = analytics.uploads_avg_inference_ms;
      const extra = (uploadsInf !== undefined && uploadsInf !== null)
        ? '\nUploads avg: ' + uploadsInf + ' ms'
        : '';
      perf.textContent = 'FPS: ' + (totals.fps ?? 0) + '\nInference: ' + (totals.inference_ms ?? 0) + ' ms' + extra;
    }
    if (analyticsFps)       analyticsFps.textContent       = totals.fps ?? 0;
    if (analyticsInference) analyticsInference.textContent = totals.inference_ms ?? 0;
    if (analyticsObjects) {
      const combined = analytics.combined_per_minute || {};
      const sum = Object.values(combined).reduce((s, v) => s + Number(v || 0), 0);
      analyticsObjects.textContent = sum.toFixed(2);
    }
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
      const error   = document.getElementById('camError-' + cam.id);

      if (fps) fps.textContent = 'FPS: ' + (cam.fps ?? 0).toFixed(1);
      if (inf) inf.textContent = 'INF: ' + (cam.inference_ms ?? 0).toFixed(1) + ' ms';

      const running = cam.running === true;
      const label   = running ? 'Running' : 'Stopped';
      const cls     = running ? 'ok' : (cam.error ? 'bad' : 'warn');
      const statusCls = 'status ' + cls;

      if (run)     { run.textContent     = label; run.className     = statusCls; }
      if (listRun) { listRun.textContent = label; listRun.className = statusCls; }
      
      if (error) {
        if (cam.error) {
          error.textContent = '⚠ ' + cam.error;
          error.style.display = 'block';
        } else {
          error.style.display = 'none';
        }
      }
    });
  } catch { /* silent */ }
}

/* ── Boot ── */
loadCameras();
renderInferenceState();
setInterval(refreshStats, 500);
refreshStats();

/* ══════════════════════════════════════════
   WEBSOCKET VIEWER
══════════════════════════════════════════ */
(function () {
  const wsModal        = document.getElementById('wsModal');
  const wsModalClose   = document.getElementById('wsModalClose');
  const wsViewBtn      = document.getElementById('wsViewBtn');
  const wsCameraSelect = document.getElementById('wsCameraSelect');
  const wsConnectBtn   = document.getElementById('wsConnectBtn');
  const wsDisconnectBtn= document.getElementById('wsDisconnectBtn');
  const wsStatus       = document.getElementById('wsStatus');
  const wsCanvas       = document.getElementById('wsCanvas');
  const wsFps          = document.getElementById('wsFps');
  const wsInf          = document.getElementById('wsInf');

  if (!wsModal) return;

  let socket        = null;
  let activeCamId   = null;

  /* Populate WS camera dropdown whenever main cameras list changes */
  function syncWsCameraSelect() {
    if (!wsCameraSelect) return;
    const prev = wsCameraSelect.value;
    wsCameraSelect.innerHTML = '';
    if (!cameras.length) {
      const opt = document.createElement('option');
      opt.textContent = 'No cameras registered';
      opt.disabled = true;
      opt.selected = true;
      wsCameraSelect.appendChild(opt);
      return;
    }
    cameras.forEach((c) => {
      const opt = document.createElement('option');
      opt.value = String(c.id);
      opt.textContent = c.label + ' (' + c.ref + ')';
      if (String(c.id) === prev) opt.selected = true;
      wsCameraSelect.appendChild(opt);
    });
    if (!wsCameraSelect.value && cameras.length) {
      wsCameraSelect.value = String(cameras[0].id);
    }
  }

  /* Open modal */
  wsViewBtn.addEventListener('click', () => {
    syncWsCameraSelect();
    wsModal.style.display = 'flex';
  });

  /* Close modal */
  wsModalClose.addEventListener('click', disconnectWs);
  wsModalClose.addEventListener('click', () => { wsModal.style.display = 'none'; });

  /* Connect */
  wsConnectBtn.addEventListener('click', () => {
    const camId = wsCameraSelect.value;
    if (!camId) { setStatus(wsStatus, 'Select a camera', 'warn'); return; }
    connectWs(camId);
  });

  /* Disconnect */
  wsDisconnectBtn.addEventListener('click', disconnectWs);

  function connectWs(camId) {
    disconnectWs();
    activeCamId = camId;
    wsConnectBtn.disabled    = true;
    wsDisconnectBtn.disabled = false;
    setStatus(wsStatus, 'Connecting…', 'warn');

    socket = io({ transports: ['websocket'] });

    socket.on('connect', () => {
      setStatus(wsStatus, 'Connected', 'ok');
      socket.emit('subscribe_camera', { camera_id: camId });
    });

    socket.on('subscribed', () => {
      setStatus(wsStatus, 'Streaming', 'ok');
    });

    socket.on('frame', (msg) => {
      if (msg.camera_id !== activeCamId) return;
      if (wsFps) wsFps.textContent = Number(msg.fps).toFixed(1);
      if (wsInf) wsInf.textContent = Number(msg.inference_ms).toFixed(1);
      const img = new Image();
      img.onload = () => {
        if (wsCanvas.width !== img.naturalWidth)  wsCanvas.width  = img.naturalWidth;
        if (wsCanvas.height !== img.naturalHeight) wsCanvas.height = img.naturalHeight;
        wsCanvas.getContext('2d').drawImage(img, 0, 0);
      };
      img.src = 'data:image/jpeg;base64,' + msg.data;
    });

    socket.on('camera_stopped', () => {
      setStatus(wsStatus, 'Camera stopped', 'warn');
    });

    socket.on('disconnect', () => {
      setStatus(wsStatus, 'Disconnected', 'bad');
      wsConnectBtn.disabled    = false;
      wsDisconnectBtn.disabled = true;
    });

    socket.on('connect_error', () => {
      setStatus(wsStatus, 'Connection error', 'bad');
      wsConnectBtn.disabled    = false;
      wsDisconnectBtn.disabled = true;
    });
  }

  function disconnectWs() {
    if (socket) {
      if (activeCamId) socket.emit('unsubscribe_camera', { camera_id: activeCamId });
      socket.disconnect();
      socket = null;
    }
    activeCamId = null;
    wsConnectBtn.disabled    = false;
    wsDisconnectBtn.disabled = true;
    if (wsStatus) setStatus(wsStatus, 'Disconnected', 'warn');
  }

  /* Keep WS dropdown in sync when cameras are added/removed */
  const _origRenderCameras = window._renderCamerasOrig || null;
  // Patch loadCameras to also sync WS dropdown
  const _origLoad = window.loadCameras;
  const origRenderFeed = typeof renderFeedGrid === 'function' ? renderFeedGrid : null;

  // Simple MutationObserver fallback — re-sync whenever the main select changes
  const mainSelect = document.getElementById('cameraSelect');
  if (mainSelect) {
    new MutationObserver(syncWsCameraSelect).observe(mainSelect, { childList: true });
  }
})();
