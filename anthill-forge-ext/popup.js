// popup.js — Anthill Forge extension UI controller
// Manages: dashboard, config, training, export, and done states.
// Communicates with the Forge server (local or LAN).

const DEFAULT_SERVER = 'http://localhost:7800';
let API_BASE = DEFAULT_SERVER + '/api';
let statusInterval = null;
let serverConnected = false;

function setServerUrl(url) {
  // Normalize: strip trailing slash and /api suffix
  url = url.replace(/\/+$/, '').replace(/\/api$/, '');
  API_BASE = url + '/api';
  chrome.storage.local.set({ forgeServerUrl: url });
  document.getElementById('cfgServerUrl').value = url;
}

// ═══════════════════════════════════════════════════════════════════════
//  INITIALIZATION — wire up all buttons and load initial state
// ═══════════════════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {
  // ─── Offline panel ──────────────────────────────────────────────
  document.getElementById('retryConnBtn').addEventListener('click', checkConnection);
  document.getElementById('helpToggleOffline').addEventListener('click', () => {
    toggleHelp('helpToggleOffline', 'helpContentOffline');
  });

  // ─── Dashboard panel ───────────────────────────────────────────
  document.getElementById('openConfigBtn').addEventListener('click', () => showPanel('config'));
  document.getElementById('quickExportBtn').addEventListener('click', startExport);
  document.getElementById('helpToggleDash').addEventListener('click', () => {
    toggleHelp('helpToggleDash', 'helpContentDash');
  });

  // ─── Config panel ──────────────────────────────────────────────
  document.getElementById('backToDashBtn').addEventListener('click', () => showPanel('dashboard'));
  document.getElementById('startTrainBtn').addEventListener('click', startTraining);

  // Config tab switching
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab));
  });

  // Update effective batch size display when inputs change
  ['cfgBatchSize', 'cfgGradAccum'].forEach(id => {
    document.getElementById(id).addEventListener('input', updateEffBatch);
  });

  // Update LoRA scaling display
  ['cfgLoraR', 'cfgLoraAlpha'].forEach(id => {
    document.getElementById(id).addEventListener('input', updateScaling);
  });

  // ─── Model download ────────────────────────────────────────────
  document.getElementById('checkModelBtn').addEventListener('click', checkModelCache);
  document.getElementById('downloadModelBtn').addEventListener('click', downloadModel);
  document.getElementById('dlStopBtn').addEventListener('click', () => {
    fetchAPI('/model/stop', { method: 'POST' });
  });

  // ─── Training panel ────────────────────────────────────────────
  document.getElementById('stopTrainBtn').addEventListener('click', confirmStopTraining);

  // ─── Export panel ──────────────────────────────────────────────
  document.getElementById('stopExportBtn').addEventListener('click', confirmStopExport);

  // ─── Done panel ────────────────────────────────────────────────
  document.getElementById('doneExportBtn').addEventListener('click', startExport);
  document.getElementById('doneNewBtn').addEventListener('click', () => showPanel('dashboard'));

  // ─── Error panel ───────────────────────────────────────────────
  document.getElementById('errorBackBtn').addEventListener('click', () => showPanel('dashboard'));

  // ─── Server URL toggle ─────────────────────────────────────
  document.getElementById('serverUrlToggle').addEventListener('click', () => {
    const bar = document.getElementById('serverUrlBar');
    bar.style.display = bar.style.display === 'none' ? 'block' : 'none';
  });
  document.getElementById('applyServerUrl').addEventListener('click', () => {
    const url = document.getElementById('cfgServerUrl').value.trim();
    if (url) {
      setServerUrl(url);
      document.getElementById('serverUrlBar').style.display = 'none';
      checkConnection();
    }
  });
  // Also apply on Enter key
  document.getElementById('cfgServerUrl').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') document.getElementById('applyServerUrl').click();
  });

  // Load saved server URL, then config, then connect
  chrome.storage.local.get(['forgeServerUrl'], (result) => {
    if (result.forgeServerUrl) {
      setServerUrl(result.forgeServerUrl);
    } else {
      document.getElementById('cfgServerUrl').value = DEFAULT_SERVER;
    }

    // Load saved config from chrome.storage
    loadSavedConfig();

    // Try connecting to server
    checkConnection();
  });
});


// ═══════════════════════════════════════════════════════════════════════
//  SERVER CONNECTION — check if forge_server.py is running
// ═══════════════════════════════════════════════════════════════════════

async function checkConnection() {
  const dot = document.getElementById('connDot');
  const text = document.getElementById('connText');

  dot.className = 'dot dot-yellow';
  text.textContent = 'Connecting...';

  try {
    const resp = await fetchAPI('/status');
    if (resp.ok) {
      serverConnected = true;
      dot.className = 'dot dot-green';
      // Show which server we're connected to
      const serverHost = API_BASE.replace('/api', '').replace('http://', '');
      text.textContent = `Connected: ${serverHost}`;

      // Check if there's an active training/export session
      const status = await resp.json();
      handleServerStatus(status);
    } else {
      throw new Error('Bad response');
    }
  } catch (e) {
    serverConnected = false;
    dot.className = 'dot dot-red';
    text.textContent = 'Server offline';
    showPanel('offline');
  }
}

// Handle the server status and show the right panel
function handleServerStatus(status) {
  switch (status.state) {
    case 'downloading':
      showPanel('download');
      updateDownloadPanel(status);
      startPolling();
      break;

    case 'training':
      showPanel('training');
      startPolling();
      break;

    case 'exporting':
      showPanel('export');
      startPolling();
      break;

    case 'done':
      showPanel('done');
      updateDonePanel(status);
      break;

    case 'error':
      showPanel('error');
      document.getElementById('errorMessage').innerHTML =
        `<strong>Error:</strong> ${status.error || 'Unknown error'}`;
      break;

    default: // idle
      showPanel('dashboard');
      loadDashboardData();
      break;
  }
}


// ═══════════════════════════════════════════════════════════════════════
//  PANEL MANAGEMENT — show/hide the appropriate UI section
// ═══════════════════════════════════════════════════════════════════════

function showPanel(name) {
  const panels = [
    'offlinePanel', 'dashboardPanel', 'configPanel',
    'downloadPanel', 'trainingPanel', 'exportPanel', 'donePanel', 'errorPanel'
  ];

  // Hide all panels
  panels.forEach(id => document.getElementById(id).classList.add('hidden'));

  // Show the requested panel
  const panelId = name + 'Panel';
  const panel = document.getElementById(panelId);
  if (panel) {
    panel.classList.remove('hidden');
  }

  // Stop polling if we're going back to a non-active state
  if (!['training', 'export', 'download'].includes(name)) {
    stopPolling();
  }
}


// ═══════════════════════════════════════════════════════════════════════
//  DASHBOARD — system stats and dataset info
// ═══════════════════════════════════════════════════════════════════════

async function loadDashboardData() {
  try {
    // Fetch system status and datasets in parallel
    const [statusResp, datasetsResp] = await Promise.all([
      fetchAPI('/status'),
      fetchAPI('/datasets'),
    ]);

    if (statusResp.ok) {
      const status = await statusResp.json();
      // Update stat boxes
      document.getElementById('statGpuName').textContent =
        status.gpu_name ? shortenGpuName(status.gpu_name) : 'No GPU';
      document.getElementById('statVram').textContent =
        status.vram_total ? `${status.vram_total.toFixed(0)}GB` : '--';
      document.getElementById('statRam').textContent =
        status.ram_total ? `${status.ram_total.toFixed(0)}GB` : '--';
    }

    if (datasetsResp.ok) {
      const data = await datasetsResp.json();
      const datasets = data.datasets || [];

      document.getElementById('statDatasets').textContent =
        datasets.length > 0 ? datasets.length : '0';

      // Build dataset list HTML
      const listEl = document.getElementById('datasetList');
      if (datasets.length === 0) {
        listEl.innerHTML = 'No JSONL datasets found.<br><span class="text-xs">Run Anthill Loom first to create training data.</span>';
      } else {
        let html = '';
        let totalPairs = 0;
        datasets.forEach(ds => {
          html += `${ds.name}: ${ds.pairs.toLocaleString()} pairs<br>`;
          totalPairs += ds.pairs;
        });
        html += `<strong>Total: ${totalPairs.toLocaleString()} training pairs</strong>`;
        listEl.innerHTML = html;
      }

      // Populate dataset dropdown in config
      const select = document.getElementById('cfgDataset');
      select.innerHTML = '<option value="all">All datasets</option>';
      datasets.forEach(ds => {
        const opt = document.createElement('option');
        opt.value = ds.name;
        opt.textContent = `${ds.name} (${ds.pairs.toLocaleString()})`;
        select.appendChild(opt);
      });
    }
  } catch (e) {
    console.error('Dashboard load error:', e);
  }
}

// Shorten GPU names for display (e.g., "NVIDIA GeForce RTX 4070 Ti Super" -> "RTX 4070 Ti S")
function shortenGpuName(name) {
  return name
    .replace('NVIDIA GeForce ', '')
    .replace('Super', 'S')
    .replace('NVIDIA ', '');
}


// ═══════════════════════════════════════════════════════════════════════
//  CONFIG — tab switching and saved preferences
// ═══════════════════════════════════════════════════════════════════════

function switchTab(tabId) {
  // Deactivate all tabs and content
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

  // Activate the clicked tab and its content
  document.querySelector(`[data-tab="${tabId}"]`).classList.add('active');
  document.getElementById(tabId).classList.add('active');
}

function updateEffBatch() {
  const batch = parseInt(document.getElementById('cfgBatchSize').value) || 1;
  const accum = parseInt(document.getElementById('cfgGradAccum').value) || 1;
  document.getElementById('cfgEffBatch').textContent = batch * accum;
}

function updateScaling() {
  const r = parseInt(document.getElementById('cfgLoraR').value) || 64;
  const alpha = parseInt(document.getElementById('cfgLoraAlpha').value) || 16;
  document.getElementById('cfgScaling').textContent = (alpha / r).toFixed(3);
}

// Save config to chrome.storage so it persists across popup opens
function saveConfig() {
  const config = {
    model: document.getElementById('cfgModel').value,
    maxLength: document.getElementById('cfgMaxLength').value,
    epochs: document.getElementById('cfgEpochs').value,
    batchSize: document.getElementById('cfgBatchSize').value,
    gradAccum: document.getElementById('cfgGradAccum').value,
    lr: document.getElementById('cfgLR').value,
    maxHours: document.getElementById('cfgMaxHours').value,
    loraR: document.getElementById('cfgLoraR').value,
    loraAlpha: document.getElementById('cfgLoraAlpha').value,
    loraDropout: document.getElementById('cfgLoraDropout').value,
    quantBits: document.getElementById('cfgQuantBits').value,
    quantType: document.getElementById('cfgQuantType').value,
    skipMerge: document.getElementById('cfgSkipMerge').value,
  };
  chrome.storage.local.set({ forgeConfig: config });
}

// Load saved config from chrome.storage
function loadSavedConfig() {
  chrome.storage.local.get(['forgeConfig'], (result) => {
    const cfg = result.forgeConfig;
    if (!cfg) return;

    if (cfg.model) document.getElementById('cfgModel').value = cfg.model;
    if (cfg.maxLength) document.getElementById('cfgMaxLength').value = cfg.maxLength;
    if (cfg.epochs) document.getElementById('cfgEpochs').value = cfg.epochs;
    if (cfg.batchSize) document.getElementById('cfgBatchSize').value = cfg.batchSize;
    if (cfg.gradAccum) document.getElementById('cfgGradAccum').value = cfg.gradAccum;
    if (cfg.lr) document.getElementById('cfgLR').value = cfg.lr;
    if (cfg.maxHours) document.getElementById('cfgMaxHours').value = cfg.maxHours;
    if (cfg.loraR) document.getElementById('cfgLoraR').value = cfg.loraR;
    if (cfg.loraAlpha) document.getElementById('cfgLoraAlpha').value = cfg.loraAlpha;
    if (cfg.loraDropout) document.getElementById('cfgLoraDropout').value = cfg.loraDropout;
    if (cfg.quantBits) document.getElementById('cfgQuantBits').value = cfg.quantBits;
    if (cfg.quantType) document.getElementById('cfgQuantType').value = cfg.quantType;
    if (cfg.skipMerge) document.getElementById('cfgSkipMerge').value = cfg.skipMerge;

    updateEffBatch();
    updateScaling();
  });
}

// Gather config values from the form into an object for the API
function gatherConfig() {
  return {
    model: document.getElementById('cfgModel').value || 'Qwen/Qwen2.5-Coder-32B-Instruct',
    max_length: parseInt(document.getElementById('cfgMaxLength').value) || 2048,
    epochs: parseInt(document.getElementById('cfgEpochs').value) || 3,
    batch_size: parseInt(document.getElementById('cfgBatchSize').value) || 1,
    grad_accum: parseInt(document.getElementById('cfgGradAccum').value) || 16,
    lr: document.getElementById('cfgLR').value || '2e-4',
    max_hours: parseInt(document.getElementById('cfgMaxHours').value) || 24,
    lora_r: parseInt(document.getElementById('cfgLoraR').value) || 64,
    lora_alpha: parseInt(document.getElementById('cfgLoraAlpha').value) || 16,
    lora_dropout: document.getElementById('cfgLoraDropout').value || '0.05',
    quant_bits: parseInt(document.getElementById('cfgQuantBits').value) || 4,
    quant_type: document.getElementById('cfgQuantType').value || 'Q8_0',
    skip_merge: document.getElementById('cfgSkipMerge').value === 'true',
  };
}


// ═══════════════════════════════════════════════════════════════════════
//  TRAINING — start, stop, and monitor training
// ═══════════════════════════════════════════════════════════════════════

async function startTraining() {
  const config = gatherConfig();

  // Save config to chrome.storage for next time
  saveConfig();

  // Confirm with user
  const msg = `Start QLoRA training?\n\n` +
    `Model: ${config.model}\n` +
    `Epochs: ${config.epochs}\n` +
    `Max time: ${config.max_hours}h\n\n` +
    `This may take several hours.`;

  if (!confirm(msg)) return;

  // Disable button while starting
  const btn = document.getElementById('startTrainBtn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner" style="width:16px;height:16px;border-width:2px;"></span> Starting...';

  try {
    const resp = await fetchAPI('/training/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });

    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.error || 'Failed to start training');
    }

    // Switch to training panel and start polling
    showPanel('training');
    startPolling();

  } catch (e) {
    alert('Error: ' + e.message);
    btn.disabled = false;
    btn.innerHTML = '&#9654; Start Training';
  }
}

function confirmStopTraining() {
  if (!confirm('Stop training? Current progress will be saved as a checkpoint.')) return;
  stopJob('training');
}

async function stopJob(type) {
  try {
    await fetchAPI(`/${type}/stop`, { method: 'POST' });
    // Let the next poll cycle update the UI
  } catch (e) {
    console.error('Stop error:', e);
  }
}


// ═══════════════════════════════════════════════════════════════════════
//  EXPORT — start GGUF conversion
// ═══════════════════════════════════════════════════════════════════════

async function startExport() {
  const quantType = document.getElementById('cfgQuantType')?.value || 'Q8_0';

  if (!confirm(`Export merged model to GGUF (${quantType})?\n\nThis may take 30-60 minutes for large models.`)) {
    return;
  }

  try {
    const resp = await fetchAPI('/export/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ quant_type: quantType }),
    });

    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.error || 'Failed to start export');
    }

    showPanel('export');
    startPolling();

  } catch (e) {
    alert('Error: ' + e.message);
  }
}

function confirmStopExport() {
  if (!confirm('Cancel GGUF export?')) return;
  stopJob('export');
}


// ═══════════════════════════════════════════════════════════════════════
//  POLLING — periodically fetch status from server
// ═══════════════════════════════════════════════════════════════════════

function startPolling() {
  if (statusInterval) return;
  statusInterval = setInterval(pollStatus, 2000);
  // Also poll immediately
  pollStatus();
}

function stopPolling() {
  if (statusInterval) {
    clearInterval(statusInterval);
    statusInterval = null;
  }
}

async function pollStatus() {
  try {
    const resp = await fetchAPI('/status');
    if (!resp.ok) throw new Error('Status fetch failed');

    const status = await resp.json();
    updateActivePanel(status);

  } catch (e) {
    // Server went away — show offline
    serverConnected = false;
    document.getElementById('connDot').className = 'dot dot-red';
    document.getElementById('connText').textContent = 'Server offline';
    stopPolling();
    showPanel('offline');
  }
}

// Route status updates to the correct panel's update function
function updateActivePanel(status) {
  // Keep connection indicator green
  document.getElementById('connDot').className = 'dot dot-green';
  document.getElementById('connText').textContent = 'Forge server running (port 7800)';

  switch (status.state) {
    case 'training':
      updateTrainingPanel(status);
      break;

    case 'exporting':
      updateExportPanel(status);
      break;

    case 'done':
      stopPolling();
      showPanel('done');
      updateDonePanel(status);
      break;

    case 'error':
      stopPolling();
      showPanel('error');
      document.getElementById('errorMessage').innerHTML =
        `<strong>Error:</strong> ${status.error || 'Unknown error'}`;
      break;

    case 'idle':
      stopPolling();
      showPanel('dashboard');
      loadDashboardData();
      break;
  }
}

// Update the training panel with live data from the server
function updateTrainingPanel(status) {
  const t = status.training || {};

  // State label
  document.getElementById('trainStateLabel').textContent =
    t.phase || 'Training...';

  // Progress bar
  const pct = t.total_steps > 0
    ? ((t.current_step || 0) / t.total_steps * 100)
    : 0;
  document.getElementById('trainProgressFill').style.width = pct + '%';

  // Status text
  document.getElementById('trainStatusText').textContent =
    t.message || `Step ${t.current_step || 0} / ${t.total_steps || '?'}`;

  // Stats row
  document.getElementById('trainStep').textContent =
    `Step: ${(t.current_step || 0).toLocaleString()}`;
  document.getElementById('trainEpoch').textContent =
    `Epoch: ${t.current_epoch || 0}/${t.total_epochs || '?'}`;
  document.getElementById('trainETA').textContent =
    t.eta ? `ETA: ${formatDuration(t.eta)}` : 'ETA: --';

  // Live metrics
  document.getElementById('metricLoss').textContent =
    t.loss != null ? t.loss.toFixed(4) : '--';
  document.getElementById('metricLR').textContent =
    t.learning_rate != null ? t.learning_rate.toExponential(2) : '--';
  document.getElementById('metricVram').textContent =
    t.vram_used != null ? `${t.vram_used.toFixed(1)} / ${t.vram_total?.toFixed(1) || '?'} GB` : '--';
  document.getElementById('metricSpeed').textContent =
    t.samples_per_second != null ? `${t.samples_per_second.toFixed(1)} samples/s` : '--';
  document.getElementById('metricElapsed').textContent =
    t.elapsed_seconds != null ? formatDuration(t.elapsed_seconds) : '--';
}

// Update the export panel with live data from the server
function updateExportPanel(status) {
  const e = status.export || {};

  document.getElementById('exportStateLabel').textContent =
    e.phase || 'Exporting...';

  const pct = e.progress || 0;
  document.getElementById('exportProgressFill').style.width = pct + '%';

  document.getElementById('exportStatusText').textContent =
    e.message || 'Converting...';
}

// Update the done panel with final results
function updateDonePanel(status) {
  const d = status.done || {};

  // If a download just finished, show the done panel briefly then go to config
  if (d.type === 'download') {
    document.getElementById('doneMessage').innerHTML =
      `Model <strong>${d.model_id || ''}</strong> downloaded successfully! Ready to train.`;
    document.getElementById('doneLoss').textContent = 'N/A';
    document.getElementById('doneTime').textContent = 'N/A';
    document.getElementById('doneModel').textContent = d.model_id || 'N/A';
    document.getElementById('doneExportBtn').classList.add('hidden');
    return;
  }

  document.getElementById('doneMessage').innerHTML =
    d.message || 'Operation complete!';

  document.getElementById('doneLoss').textContent =
    d.final_loss != null ? d.final_loss.toFixed(4) : 'N/A';
  document.getElementById('doneTime').textContent =
    d.elapsed_seconds != null ? formatDuration(d.elapsed_seconds) : 'N/A';
  document.getElementById('doneModel').textContent =
    d.model || 'N/A';

  // Show/hide export button based on whether we just finished training (vs export)
  document.getElementById('doneExportBtn').classList.toggle('hidden', d.type === 'export');
}


// ═══════════════════════════════════════════════════════════════════════
//  HELPERS
// ═══════════════════════════════════════════════════════════════════════

// Fetch wrapper that targets the local Forge server
function fetchAPI(path, options = {}) {
  return fetch(API_BASE + path, {
    ...options,
    signal: AbortSignal.timeout(5000),
  });
}

// Format seconds into a human-readable duration (e.g. "2h 15m" or "45m 30s")
function formatDuration(seconds) {
  if (seconds == null || seconds < 0) return '--';
  seconds = Math.round(seconds);

  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) {
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return `${m}m ${s}s`;
  }
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  return `${h}h ${m}m`;
}

// Toggle help sections (shared pattern from Spider/Collector)
function toggleHelp(toggleId, contentId) {
  const toggle = document.getElementById(toggleId);
  const content = document.getElementById(contentId);
  toggle.classList.toggle('open');
  content.classList.toggle('open');
}


// ═══════════════════════════════════════════════════════════════════════
//  MODEL DOWNLOAD — check cache and download from HuggingFace
// ═══════════════════════════════════════════════════════════════════════

async function checkModelCache() {
  const modelId = document.getElementById('cfgModel').value.trim();
  const statusDiv = document.getElementById('modelCacheStatus');

  if (!modelId) {
    statusDiv.style.display = 'block';
    statusDiv.style.background = 'rgba(239,68,68,0.2)';
    statusDiv.textContent = 'Enter a model ID first';
    return;
  }

  statusDiv.style.display = 'block';
  statusDiv.style.background = 'rgba(217,119,6,0.2)';
  statusDiv.textContent = 'Checking cache...';

  try {
    const resp = await fetchAPI(`/model/check?id=${encodeURIComponent(modelId)}`);
    const data = await resp.json();

    if (data.cached) {
      statusDiv.style.background = 'rgba(34,197,94,0.2)';
      statusDiv.innerHTML = `<strong style="color:#4ade80;">Cached</strong> — ${data.size_gb} GB on disk. Ready to train!`;
    } else {
      statusDiv.style.background = 'rgba(251,191,36,0.2)';
      statusDiv.innerHTML = `<strong style="color:#fbbf24;">Not cached</strong> — click Download to fetch from HuggingFace.`;
    }
  } catch (e) {
    statusDiv.style.background = 'rgba(239,68,68,0.2)';
    statusDiv.textContent = 'Error checking cache: ' + e.message;
  }
}

async function downloadModel() {
  const modelId = document.getElementById('cfgModel').value.trim();
  if (!modelId) {
    alert('Enter a model ID first (e.g. Qwen/Qwen2.5-Coder-32B-Instruct)');
    return;
  }
  if (!modelId.includes('/')) {
    alert('Model ID should be in format: org/model\n\nExample: Qwen/Qwen2.5-Coder-32B-Instruct');
    return;
  }

  if (!confirm(`Download "${modelId}" from HuggingFace?\n\nLarge models can be 30-70 GB. Make sure you have enough disk space.`)) {
    return;
  }

  try {
    const resp = await fetchAPI('/model/download', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_id: modelId }),
    });

    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.error || 'Failed to start download');
    }

    showPanel('download');
    document.getElementById('dlModelName').textContent = modelId;
    startPolling();
  } catch (e) {
    alert('Error: ' + e.message);
  }
}

function updateDownloadPanel(status) {
  const dl = status.download || {};
  document.getElementById('dlModelName').textContent = dl.model_id || '';
  document.getElementById('dlStatusText').textContent = dl.message || dl.phase || 'Downloading...';

  // Estimate progress from files if available
  const fill = document.getElementById('dlProgressFill');
  if (dl.files_total > 0 && dl.files_done > 0) {
    const pct = Math.round((dl.files_done / dl.files_total) * 100);
    fill.style.width = pct + '%';
  } else if (dl.files_done > 0) {
    // Indeterminate — pulse between 10-50%
    fill.style.width = Math.min(50, 10 + dl.files_done * 3) + '%';
  }
}
