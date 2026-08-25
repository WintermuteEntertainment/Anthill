// background.js — Anthill Forge service worker
// Runs in the background to monitor training/export status and send
// desktop notifications when long-running jobs complete. The popup
// handles most of the UI logic; this worker just watches for state
// transitions and notifies the user.

const DEFAULT_SERVER = 'http://localhost:7800';
let API_BASE = DEFAULT_SERVER + '/api';

// Track the last known state so we can detect transitions
let lastKnownState = 'idle';
let watchInterval = null;

// Load saved server URL
chrome.storage.local.get(['forgeServerUrl'], (result) => {
  if (result.forgeServerUrl) {
    API_BASE = result.forgeServerUrl.replace(/\/+$/, '') + '/api';
  }
});

// Listen for URL changes from the popup
chrome.storage.onChanged.addListener((changes) => {
  if (changes.forgeServerUrl) {
    API_BASE = changes.forgeServerUrl.newValue.replace(/\/+$/, '') + '/api';
    console.log('Forge server URL updated:', API_BASE);
  }
});


// ═══════════════════════════════════════════════════════════════════════
//  SERVICE WORKER LIFECYCLE
// ═══════════════════════════════════════════════════════════════════════

// On install/update, start watching immediately
chrome.runtime.onInstalled.addListener(() => {
  console.log('Anthill Forge extension installed');
  startWatching();
});

// Also start watching whenever the service worker wakes up
startWatching();


// ═══════════════════════════════════════════════════════════════════════
//  STATE WATCHER — polls server and fires notifications on transitions
// ═══════════════════════════════════════════════════════════════════════

function startWatching() {
  // Don't double-start
  if (watchInterval) return;

  // Poll every 10 seconds (light touch — just watching for state changes)
  watchInterval = setInterval(checkForStateChange, 10000);

  // Also check right now
  checkForStateChange();
}

async function checkForStateChange() {
  try {
    const resp = await fetch(API_BASE + '/status', {
      signal: AbortSignal.timeout(3000),
    });

    if (!resp.ok) return;
    const status = await resp.json();
    const newState = status.state;

    // Detect transitions and send notifications
    if (newState !== lastKnownState) {
      handleStateTransition(lastKnownState, newState, status);
      lastKnownState = newState;
    }
  } catch {
    // Server offline — not an error, just means Forge isn't running
  }
}

// Send a desktop notification when a job finishes or errors out
function handleStateTransition(from, to, status) {
  // Training completed
  if (from === 'training' && to === 'done') {
    const d = status.done || {};
    chrome.notifications.create('forge-train-done', {
      type: 'basic',
      iconUrl: 'icon.png',
      title: 'Anthill Forge — Training Complete',
      message: d.message || 'QLoRA training has finished.',
    });
  }

  // Export completed
  if (from === 'exporting' && to === 'done') {
    const d = status.done || {};
    chrome.notifications.create('forge-export-done', {
      type: 'basic',
      iconUrl: 'icon.png',
      title: 'Anthill Forge — Export Complete',
      message: d.message || 'GGUF export has finished.',
    });
  }

  // Error occurred
  if (to === 'error') {
    chrome.notifications.create('forge-error', {
      type: 'basic',
      iconUrl: 'icon.png',
      title: 'Anthill Forge — Error',
      message: status.error || 'An error occurred during the operation.',
    });
  }
}


// ═══════════════════════════════════════════════════════════════════════
//  MESSAGE HANDLING — respond to popup requests
// ═══════════════════════════════════════════════════════════════════════

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  // The popup can ask the background worker for the current state
  // (useful if the popup opens and wants to skip its own initial check)
  if (message.action === 'getLastState') {
    sendResponse({ state: lastKnownState });
    return true;
  }
});
