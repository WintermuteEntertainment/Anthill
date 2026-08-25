// popup.js — Anthill Loom extension UI controller
// Manages: dashboard, processing, done, and error states.
// Communicates with the local Loom server at localhost:7801.

const API_BASE = 'http://localhost:7801/api';
let statusInterval = null;
let serverConnected = false;

// ═══════════════════════════════════════════════════════════════════════
//  INITIALIZATION — wire up buttons and load initial state
// ═══════════════════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {
  // ─── Offline panel ──────────────────────────────────────────────
  document.getElementById('retryConnBtn').addEventListener('click', checkConnection);
  document.getElementById('helpToggleOffline').addEventListener('click', () => {
    toggleHelp('helpToggleOffline', 'helpContentOffline');
  });

  // ─── Dashboard panel ───────────────────────────────────────────
  document.getElementById('processBtn').addEventListener('click', processSelected);
  document.getElementById('processLatestBtn').addEventListener('click', processLatest);
  document.getElementById('helpToggleDash').addEventListener('click', () => {
    toggleHelp('helpToggleDash', 'helpContentDash');
  });

  // Enable/disable process button based on file selection
  document.getElementById('inputFileSelect').addEventListener('change', (e) => {
    document.getElementById('processBtn').disabled = !e.target.value;
  });

  // ─── Done panel ────────────────────────────────────────────────
  document.getElementById('doneBackBtn').addEventListener('click', () => {
    showPanel('dashboard');
    loadDashboardData();
  });

  // ─── Error panel ───────────────────────────────────────────────
  document.getElementById('errorBackBtn').addEventListener('click', () => {
    showPanel('dashboard');
    loadDashboardData();
  });

  // Try connecting to server
  checkConnection();
});


// ═══════════════════════════════════════════════════════════════════════
//  SERVER CONNECTION
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
      text.textContent = 'Loom server running (port 7801)';

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

// Route to the right panel based on server state
function handleServerStatus(status) {
  switch (status.state) {
    case 'processing':
      showPanel('processing');
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
//  PANEL MANAGEMENT
// ═══════════════════════════════════════════════════════════════════════

function showPanel(name) {
  const panels = [
    'offlinePanel', 'dashboardPanel', 'processingPanel',
    'donePanel', 'errorPanel'
  ];

  panels.forEach(id => document.getElementById(id).classList.add('hidden'));

  const panel = document.getElementById(name + 'Panel');
  if (panel) panel.classList.remove('hidden');

  if (name !== 'processing') stopPolling();
}


// ═══════════════════════════════════════════════════════════════════════
//  DASHBOARD — load file lists and stats
// ═══════════════════════════════════════════════════════════════════════

async function loadDashboardData() {
  try {
    const resp = await fetchAPI('/status');
    if (!resp.ok) return;

    const data = await resp.json();

    // ─── Stats boxes ──────────────────────────────────────────────
    const raw = data.raw_files || [];
    const output = data.output_files || [];

    document.getElementById('statRawFiles').textContent = raw.length;
    document.getElementById('statConversations').textContent =
      data.total_conversations != null ? data.total_conversations.toLocaleString() : '--';

    // Show clean pair count from the latest output file
    if (output.length > 0) {
      const latest = output[output.length - 1];
      document.getElementById('statCleanPairs').textContent =
        latest.pairs != null ? latest.pairs.toLocaleString() : '--';
    } else {
      document.getElementById('statCleanPairs').textContent = '0';
    }

    // ─── Raw file list ────────────────────────────────────────────
    const rawListEl = document.getElementById('rawFileList');
    const selectEl = document.getElementById('inputFileSelect');

    // Reset the dropdown (keep the placeholder)
    selectEl.innerHTML = '<option value="">Select a raw JSON file...</option>';

    if (raw.length === 0) {
      rawListEl.innerHTML = 'No raw JSON files found.<br>' +
        '<span class="text-xs">Run Anthill Spider to export conversations first.</span>';
    } else {
      let html = '';
      raw.forEach(f => {
        html += `<div class="file-item">` +
          `<span class="file-name" title="${f.name}">${f.name}</span>` +
          `<span class="file-meta">${f.size_mb} MB` +
          `${f.conversations ? ' / ' + f.conversations + ' convos' : ''}</span>` +
          `</div>`;

        // Add to dropdown
        const opt = document.createElement('option');
        opt.value = f.name;
        opt.textContent = `${f.name} (${f.size_mb} MB)`;
        selectEl.appendChild(opt);
      });
      rawListEl.innerHTML = html;
    }

    // ─── Output file list ─────────────────────────────────────────
    const outListEl = document.getElementById('outputFileList');
    if (output.length === 0) {
      outListEl.innerHTML = 'No processed files yet.<br>' +
        '<span class="text-xs">Select a raw file above and click Extract.</span>';
    } else {
      let html = '';
      output.forEach(f => {
        html += `<div class="file-item">` +
          `<span class="file-name" title="${f.name}">${f.name}</span>` +
          `<span class="file-meta">${f.pairs != null ? f.pairs.toLocaleString() + ' pairs' : f.size_mb + ' MB'}</span>` +
          `</div>`;
      });
      outListEl.innerHTML = html;
    }

    // Reset process button state
    document.getElementById('processBtn').disabled = !selectEl.value;

  } catch (e) {
    console.error('Dashboard load error:', e);
  }
}


// ═══════════════════════════════════════════════════════════════════════
//  PROCESSING — start the extraction pipeline
// ═══════════════════════════════════════════════════════════════════════

async function processSelected() {
  const filename = document.getElementById('inputFileSelect').value;
  if (!filename) return;

  if (!confirm(`Process "${filename}"?\n\nThis will extract prompt/response pairs, deduplicate, and filter.`)) {
    return;
  }

  await startProcessing(filename);
}

async function processLatest() {
  if (!confirm('Process the most recent Spider export?\n\nThis will find the latest raw JSON and extract training pairs from it.')) {
    return;
  }

  await startProcessing(null); // null = server picks the latest
}

async function startProcessing(filename) {
  const btn = filename
    ? document.getElementById('processBtn')
    : document.getElementById('processLatestBtn');

  btn.disabled = true;
  btn.innerHTML = '<span class="spinner" style="width:16px;height:16px;border-width:2px;"></span> Starting...';

  try {
    const body = filename ? { filename } : { latest: true };

    const resp = await fetchAPI('/process', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.error || 'Failed to start processing');
    }

    showPanel('processing');
    startPolling();

  } catch (e) {
    alert('Error: ' + e.message);
    btn.disabled = false;
    btn.innerHTML = filename
      ? '&#9654; Extract &amp; Clean Pairs'
      : '&#x1F504; Process Latest Export';
  }
}


// ═══════════════════════════════════════════════════════════════════════
//  POLLING — monitor processing progress
// ═══════════════════════════════════════════════════════════════════════

function startPolling() {
  if (statusInterval) return;
  statusInterval = setInterval(pollStatus, 1000); // fast poll — Loom runs quickly
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

    // Keep connection indicator green
    document.getElementById('connDot').className = 'dot dot-green';
    document.getElementById('connText').textContent = 'Loom server running (port 7801)';

    switch (status.state) {
      case 'processing':
        updateProcessingPanel(status);
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
  } catch (e) {
    serverConnected = false;
    document.getElementById('connDot').className = 'dot dot-red';
    document.getElementById('connText').textContent = 'Server offline';
    stopPolling();
    showPanel('offline');
  }
}

function updateProcessingPanel(status) {
  const p = status.processing || {};

  document.getElementById('processStateLabel').textContent =
    p.phase || 'Processing...';

  const pct = p.progress || 0;
  document.getElementById('processProgressFill').style.width = pct + '%';

  document.getElementById('processStatusText').textContent =
    p.message || 'Working...';

  document.getElementById('processDetail').textContent =
    p.detail || 'Extracting pairs...';
}

function updateDonePanel(status) {
  const d = status.done || {};

  document.getElementById('doneMessage').innerHTML =
    d.message || 'Processing complete!';

  document.getElementById('resConversations').textContent =
    d.conversations != null ? d.conversations.toLocaleString() : '--';
  document.getElementById('resMessages').textContent =
    d.total_messages != null ? d.total_messages.toLocaleString() : '--';
  document.getElementById('resRawPairs').textContent =
    d.raw_pairs != null ? d.raw_pairs.toLocaleString() : '--';
  document.getElementById('resCleanPairs').textContent =
    d.clean_pairs != null ? d.clean_pairs.toLocaleString() : '--';
  document.getElementById('resDupes').textContent =
    d.duplicates_removed != null ? d.duplicates_removed.toLocaleString() : '--';
  document.getElementById('resOutputFile').textContent =
    d.output_file || '--';
}


// ═══════════════════════════════════════════════════════════════════════
//  HELPERS
// ═══════════════════════════════════════════════════════════════════════

function fetchAPI(path, options = {}) {
  return fetch(API_BASE + path, {
    ...options,
    signal: AbortSignal.timeout(5000),
  });
}

function toggleHelp(toggleId, contentId) {
  const toggle = document.getElementById(toggleId);
  const content = document.getElementById(contentId);
  toggle.classList.toggle('open');
  content.classList.toggle('open');
}
