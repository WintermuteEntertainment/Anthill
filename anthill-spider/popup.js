// popup.js — Anthill Spider v2 UI
// Manages: idle, running, paused, done states + pick mode + auto-resume

let statusInterval = null;
let pickModeActive = false;

document.addEventListener('DOMContentLoaded', () => {
  // ─── Idle panel ────────────────────────────────────────────────────
  document.getElementById('findAllBtn').addEventListener('click', findAllConversations);
  document.getElementById('startBtn').addEventListener('click', startExport);
  document.getElementById('pickBtn').addEventListener('click', enterPickMode);
  document.getElementById('cancelPickBtn').addEventListener('click', cancelPickMode);

  // ─── Running panel ─────────────────────────────────────────────────
  document.getElementById('pauseBtn').addEventListener('click', () => sendAction('pause'));
  document.getElementById('resumeRunBtn').addEventListener('click', () => sendAction('resume'));
  document.getElementById('stopBtn').addEventListener('click', confirmStop);
  document.getElementById('saveMidBtn').addEventListener('click', () => sendAction('downloadNow'));

  // ─── Done panel ────────────────────────────────────────────────────
  document.getElementById('downloadBtn').addEventListener('click', () => sendAction('downloadNow'));
  document.getElementById('newSessionBtn').addEventListener('click', () => {
    sendAction('reset');
    setTimeout(refreshState, 300);
  });

  // ─── Resume banner ─────────────────────────────────────────────────
  document.getElementById('resumeBtn').addEventListener('click', () => {
    sendAction('resume');
    setTimeout(refreshState, 300);
  });
  document.getElementById('discardBtn').addEventListener('click', () => {
    sendAction('reset');
    setTimeout(refreshState, 300);
  });
  document.getElementById('downloadPartialBtn').addEventListener('click', () => sendAction('downloadNow'));

  // ─── Help toggle ───────────────────────────────────────────────────
  document.getElementById('helpToggle').addEventListener('click', toggleHelp);

  // Load saved limit
  chrome.storage.local.get(['spiderLimit'], (result) => {
    if (result.spiderLimit) {
      document.getElementById('limitInput').value = result.spiderLimit;
    }
  });

  // Save limit on change
  document.getElementById('limitInput').addEventListener('change', (e) => {
    const val = parseInt(e.target.value) || 0;
    chrome.storage.local.set({ spiderLimit: val });
  });

  // Listen for discovery progress from content script (relayed through background)
  chrome.runtime.onMessage.addListener((msg) => {
    if (msg.action === 'discoveryProgress') {
      const statusEl = document.getElementById('discoveryScanStatus');
      statusEl.classList.remove('hidden');
      statusEl.textContent = '';
      const spinner = document.createElement('span');
      spinner.className = 'spinner';
      spinner.style.cssText = 'width:12px;height:12px;border-width:2px;vertical-align:middle;margin-right:6px;';
      statusEl.appendChild(spinner);
      statusEl.appendChild(document.createTextNode(msg.message));
    }
  });

  // Initial state check
  refreshState();
});


// ═══════════════════════════════════════════════════════════════════════
//  STATE MANAGEMENT
// ═══════════════════════════════════════════════════════════════════════

function refreshState() {
  chrome.runtime.sendMessage({ action: 'getStatus' }, (status) => {
    if (chrome.runtime.lastError) {
      console.error('Status error:', chrome.runtime.lastError);
      showPanel('idle');
      return;
    }
    updateUI(status);
  });
}

function updateUI(status) {
  if (!status) {
    showPanel('idle');
    return;
  }

  const { state, total, completed, failed, currentIndex, startTime, hasSavedData } = status;

  switch (state) {
    case 'running':
      showPanel('running');
      document.getElementById('pauseBtn').classList.remove('hidden');
      document.getElementById('resumeRunBtn').classList.add('hidden');
      document.getElementById('stateLabel').textContent = 'Scraping...';
      updateProgress(total, completed, failed, startTime);
      startPolling();
      break;

    case 'paused':
      // If popup just opened and we detect a paused session, show resume banner
      if (!statusInterval) {
        showPanel('idle');
        showResumeBanner(total, completed, failed);
      } else {
        // Paused while running — stay on running panel with resume button
        showPanel('running');
        document.getElementById('pauseBtn').classList.add('hidden');
        document.getElementById('resumeRunBtn').classList.remove('hidden');
        document.getElementById('stateLabel').textContent = 'Paused';
        updateProgress(total, completed, failed, startTime);
      }
      break;

    case 'done':
      showPanel('done');
      stopPolling();
      const partial = completed < total;
      const msg = partial
        ? `Scraped <strong>${completed}</strong> of ${total} conversations (${failed} failed). Partial export available.`
        : `Successfully scraped all <strong>${completed}</strong> conversations!${failed ? ` (${failed} failed)` : ''}`;
      document.getElementById('doneMessage').innerHTML = msg;
      break;

    default: // idle
      showPanel('idle');
      stopPolling();
      break;
  }
}

function showPanel(name) {
  const panels = ['idlePanel', 'runningPanel', 'donePanel', 'pickActivePanel', 'singleScrapePanel'];
  panels.forEach(id => document.getElementById(id).classList.add('hidden'));
  document.getElementById('resumeBanner').classList.add('hidden');

  switch (name) {
    case 'idle':      document.getElementById('idlePanel').classList.remove('hidden'); break;
    case 'running':   document.getElementById('runningPanel').classList.remove('hidden'); break;
    case 'done':      document.getElementById('donePanel').classList.remove('hidden'); break;
    case 'pick':      document.getElementById('pickActivePanel').classList.remove('hidden'); break;
    case 'single':    document.getElementById('singleScrapePanel').classList.remove('hidden'); break;
  }
}

function showResumeBanner(total, completed, failed) {
  const banner = document.getElementById('resumeBanner');
  const remaining = total - completed - failed;
  document.getElementById('resumeInfo').textContent =
    `${completed} of ${total} done, ${remaining} remaining${failed ? `, ${failed} failed` : ''}.`;
  banner.classList.remove('hidden');
}

function updateProgress(total, completed, failed, startTime) {
  const pct = total > 0 ? ((completed + failed) / total * 100) : 0;
  document.getElementById('progressFill').style.width = pct + '%';

  const remaining = total - completed - failed;
  document.getElementById('statusText').textContent =
    `${completed + failed} / ${total} processed (${remaining} remaining)`;

  document.getElementById('statsDone').textContent = `Done: ${completed}`;
  document.getElementById('statsFailed').textContent = `Failed: ${failed}`;

  // ETA calculation
  if (completed > 0 && startTime) {
    const elapsed = (Date.now() - new Date(startTime).getTime()) / 1000;
    const avgPer = elapsed / completed;
    const etaSec = remaining * avgPer;

    if (etaSec > 60) {
      const min = Math.ceil(etaSec / 60);
      document.getElementById('statsETA').textContent = `ETA: ~${min} min`;
    } else {
      document.getElementById('statsETA').textContent = `ETA: ~${Math.ceil(etaSec)}s`;
    }
  } else {
    document.getElementById('statsETA').textContent = 'ETA: calculating...';
  }
}

function startPolling() {
  if (statusInterval) return;
  statusInterval = setInterval(refreshState, 1500);
}

function stopPolling() {
  if (statusInterval) {
    clearInterval(statusInterval);
    statusInterval = null;
  }
}


// ═══════════════════════════════════════════════════════════════════════
//  BATCH EXPORT
// ═══════════════════════════════════════════════════════════════════════

async function startExport() {
  const limit = parseInt(document.getElementById('limitInput').value) || 0;

  // Check we're on ChatGPT
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.url || (!tab.url.includes('chatgpt.com') && !tab.url.includes('chat.openai.com'))) {
    alert('Please open ChatGPT.com in the active tab first.');
    return;
  }

  // Show loading state
  document.getElementById('startBtn').disabled = true;
  document.getElementById('startBtn').innerHTML = '<span class="spinner" style="width:16px;height:16px;border-width:2px;"></span> Scanning sidebar...';

  try {
    await ensureContentScript(tab.id);

    // Extract sidebar links
    const response = await tabMessage(tab.id, { action: 'extractLinks', limit });

    if (!response?.ok || !response.conversations?.length) {
      throw new Error(response?.error || 'No conversations found. Make sure conversations are visible in the sidebar.');
    }

    const conversations = response.conversations;
    const totalFound = response.total || conversations.length;

    // Build confirmation message
    const avgSec = 20;
    const estSec = conversations.length * avgSec;
    const estMin = Math.ceil(estSec / 60);
    const limitNote = limit > 0 ? ` (limited to last ${limit} of ${totalFound} found)` : '';

    const proceed = confirm(
      `Found ${conversations.length} conversations${limitNote}.\n\n` +
      `Estimated time: ~${estMin > 1 ? estMin + ' minutes' : estSec + ' seconds'}.\n\n` +
      `Tabs will open in the background. Don't close Chrome.\n\nContinue?`
    );

    if (!proceed) {
      resetStartBtn();
      return;
    }

    // Launch scraping
    chrome.runtime.sendMessage({
      action: 'startScraping',
      conversations,
      limit,
    });

    // Switch to running panel
    setTimeout(refreshState, 500);

  } catch (e) {
    alert('Error: ' + e.message);
    resetStartBtn();
  }
}

function resetStartBtn() {
  const btn = document.getElementById('startBtn');
  btn.disabled = false;
  btn.innerHTML = '&#x1F680; Start Export';
}


// ═══════════════════════════════════════════════════════════════════════
//  PICK MODE — single conversation export
// ═══════════════════════════════════════════════════════════════════════

async function enterPickMode() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.url || (!tab.url.includes('chatgpt.com') && !tab.url.includes('chat.openai.com'))) {
    alert('Please open ChatGPT.com in the active tab first.');
    return;
  }

  await ensureContentScript(tab.id);

  // Enable pick mode in content script
  await tabMessage(tab.id, { action: 'enablePickMode' });
  pickModeActive = true;
  showPanel('pick');

  // Listen for the pick result from background
  const pickListener = (msg) => {
    if (msg.action === 'pickConversation' || msg.action === 'pickCancelled') {
      // These are handled by background.js, but we also listen here for UI updates
    }
  };

  // Poll for pick mode completion (the popup can't receive messages from background
  // directly, so we'll just check periodically and also the user can cancel)
  const pickPoll = setInterval(async () => {
    if (!pickModeActive) {
      clearInterval(pickPoll);
      return;
    }
    // Check if content script still has pick mode active
    try {
      const resp = await tabMessage(tab.id, { action: 'ping' });
      // If we get here, content script is still alive
    } catch {
      // Tab closed or navigated away
      pickModeActive = false;
      clearInterval(pickPoll);
      showPanel('idle');
    }
  }, 2000);
}

async function cancelPickMode() {
  pickModeActive = false;
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (tab?.id) {
    try {
      await tabMessage(tab.id, { action: 'disablePickMode' });
    } catch {}
  }
  showPanel('idle');
}

// Listen for single-scrape results from background
chrome.runtime.onMessage.addListener((msg) => {
  if (msg.action === 'pickConversation') {
    // Background is handling the scrape — show loading
    pickModeActive = false;
    showPanel('single');
    document.getElementById('singleScrapeText').textContent =
      `Scraping "${msg.conversation?.title || 'conversation'}"...`;

    // Timeout: if scrape takes more than 60s, revert to idle
    setTimeout(() => {
      const panel = document.getElementById('singleScrapePanel');
      if (!panel.classList.contains('hidden')) {
        document.getElementById('singleScrapeText').textContent = 'Scrape timed out. Try again.';
        setTimeout(() => showPanel('idle'), 3000);
      }
    }, 60000);
  }
  if (msg.action === 'pickCancelled') {
    pickModeActive = false;
    showPanel('idle');
  }
});


// ═══════════════════════════════════════════════════════════════════════
//  FIND ALL CONVERSATIONS
// ═══════════════════════════════════════════════════════════════════════

async function findAllConversations() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.url || (!tab.url.includes('chatgpt.com') && !tab.url.includes('chat.openai.com'))) {
    alert('Please open ChatGPT.com in the active tab first.');
    return;
  }

  const btn = document.getElementById('findAllBtn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner" style="width:14px;height:14px;border-width:2px;"></span> Scanning sidebar...';

  const statusEl = document.getElementById('discoveryScanStatus');
  statusEl.classList.remove('hidden');
  statusEl.textContent = 'Starting sidebar scan...';

  try {
    await ensureContentScript(tab.id);

    const response = await tabMessage(tab.id, { action: 'findAllConversations' });

    if (!response?.ok) {
      throw new Error(response?.error || 'Failed to scan sidebar.');
    }

    const total = response.total || 0;

    // Show discovery result
    document.getElementById('discoveryCount').textContent = total;
    document.getElementById('discoveryResult').classList.remove('hidden');

    // Update the limit input hint
    statusEl.textContent = `Sidebar fully loaded. Set a limit below or export all ${total}.`;

  } catch (e) {
    statusEl.textContent = `Error: ${e.message}`;
  }

  btn.disabled = false;
  btn.innerHTML = '&#x1F50D; Find All Conversations';
}


// ═══════════════════════════════════════════════════════════════════════
//  HELP TOGGLE
// ═══════════════════════════════════════════════════════════════════════

function toggleHelp() {
  const toggle = document.getElementById('helpToggle');
  const content = document.getElementById('helpContent');
  toggle.classList.toggle('open');
  content.classList.toggle('open');
}


// ═══════════════════════════════════════════════════════════════════════
//  STOP CONFIRMATION
// ═══════════════════════════════════════════════════════════════════════

function confirmStop() {
  const msg = 'Stop scraping? All progress so far will be saved and downloaded.';
  if (confirm(msg)) {
    sendAction('stop');
    stopPolling();
    setTimeout(refreshState, 1000);
  }
}


// ═══════════════════════════════════════════════════════════════════════
//  HELPERS
// ═══════════════════════════════════════════════════════════════════════

function sendAction(action) {
  chrome.runtime.sendMessage({ action }, () => {
    if (chrome.runtime.lastError) {
      console.error('sendAction error:', chrome.runtime.lastError);
    }
  });
}

function tabMessage(tabId, message) {
  return new Promise((resolve, reject) => {
    chrome.tabs.sendMessage(tabId, message, (response) => {
      if (chrome.runtime.lastError) {
        reject(new Error(chrome.runtime.lastError.message));
      } else {
        resolve(response);
      }
    });
  });
}

async function ensureContentScript(tabId) {
  try {
    const resp = await tabMessage(tabId, { action: 'ping' });
    if (resp?.ok) return;
  } catch {}
  await chrome.scripting.executeScript({
    target: { tabId },
    files: ['content.js'],
  });
  await sleep(500);
}

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}
