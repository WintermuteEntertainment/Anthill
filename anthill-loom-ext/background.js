// background.js — Anthill Loom service worker
// Monitors the Loom server for state changes and sends desktop
// notifications when processing completes. Loom jobs are fast
// (seconds, not hours) so this is mainly useful if the user
// starts a job and switches tabs.

const API_BASE = 'http://localhost:7801/api';

// Track last known state for transition detection
let lastKnownState = 'idle';
let watchInterval = null;


// ═══════════════════════════════════════════════════════════════════════
//  SERVICE WORKER LIFECYCLE
// ═══════════════════════════════════════════════════════════════════════

chrome.runtime.onInstalled.addListener(() => {
  console.log('Anthill Loom extension installed');
  startWatching();
});

// Start watching whenever the service worker wakes up
startWatching();


// ═══════════════════════════════════════════════════════════════════════
//  STATE WATCHER — poll server, fire notifications on transitions
// ═══════════════════════════════════════════════════════════════════════

function startWatching() {
  if (watchInterval) return;

  // Poll every 5 seconds (Loom jobs are quick, so this is responsive enough)
  watchInterval = setInterval(checkForStateChange, 5000);
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

    if (newState !== lastKnownState) {
      handleStateTransition(lastKnownState, newState, status);
      lastKnownState = newState;
    }
  } catch {
    // Server offline — that's fine, Loom just isn't running
  }
}

function handleStateTransition(from, to, status) {
  // Processing completed
  if (from === 'processing' && to === 'done') {
    const d = status.done || {};
    const pairs = d.clean_pairs != null ? d.clean_pairs.toLocaleString() : '?';
    chrome.notifications.create('loom-done', {
      type: 'basic',
      iconUrl: 'icon.png',
      title: 'Anthill Loom — Processing Complete',
      message: `Extracted ${pairs} clean training pairs. Ready for Forge.`,
    });
  }

  // Error occurred
  if (to === 'error') {
    chrome.notifications.create('loom-error', {
      type: 'basic',
      iconUrl: 'icon.png',
      title: 'Anthill Loom — Error',
      message: status.error || 'An error occurred during processing.',
    });
  }
}


// ═══════════════════════════════════════════════════════════════════════
//  MESSAGE HANDLING
// ═══════════════════════════════════════════════════════════════════════

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === 'getLastState') {
    sendResponse({ state: lastKnownState });
    return true;
  }
});
