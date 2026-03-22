// background.js — Anthill Spider v2 Service Worker
// Orchestrates batch scraping with pause/resume/partial-save

console.log('[Spider] Background script loaded at', new Date().toISOString());

// ═══════════════════════════════════════════════════════════════════════
//  SESSION STATE — persisted to chrome.storage.local
// ═══════════════════════════════════════════════════════════════════════

const DEFAULT_SESSION = {
  state: 'idle',          // idle | running | paused | done
  total: 0,
  completed: 0,
  failed: 0,
  conversations: [],      // scraped data so far
  allConversations: [],   // full list of conversations to scrape
  startTime: null,
  endTime: null,
  currentIndex: 0,
  processingTab: null,
  limit: 0,               // 0 = all, >0 = last N
};

let session = { ...DEFAULT_SESSION };
let activeTimers = new Map();

// Keep service worker alive during scraping
let keepAlive = null;
function startKeepAlive() {
  if (keepAlive) return;
  keepAlive = setInterval(() => {}, 20000);
}
function stopKeepAlive() {
  if (keepAlive) { clearInterval(keepAlive); keepAlive = null; }
}

// Load saved session on startup (enables auto-resume)
chrome.storage.local.get(['spiderSession'], (result) => {
  if (result.spiderSession) {
    session = { ...DEFAULT_SESSION, ...result.spiderSession };
    console.log(`[Spider] Restored session: state=${session.state}, ${session.completed}/${session.total}`);

    // If it was running when Chrome closed, mark as paused so user can resume
    if (session.state === 'running') {
      session.state = 'paused';
      session.processingTab = null;
      saveSession();
      console.log('[Spider] Session was running — marked as paused for auto-resume');
    }
  }
});

function saveSession() {
  chrome.storage.local.set({ spiderSession: session });
}

// ═══════════════════════════════════════════════════════════════════════
//  MESSAGE ROUTER
// ═══════════════════════════════════════════════════════════════════════

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  try {
    switch (msg.action) {
      case 'startScraping':
        startScraping(msg.conversations, msg.limit);
        sendResponse({ ok: true });
        break;

      case 'getStatus':
        sendResponse({
          state: session.state,
          total: session.total,
          completed: session.completed,
          failed: session.failed,
          currentIndex: session.currentIndex,
          startTime: session.startTime,
          limit: session.limit,
          hasSavedData: session.conversations.length > 0,
        });
        break;

      case 'pause':
        pauseScraping();
        sendResponse({ ok: true });
        break;

      case 'resume':
        resumeScraping();
        sendResponse({ ok: true });
        break;

      case 'stop':
        stopScraping(true);  // true = save partial
        sendResponse({ ok: true });
        break;

      case 'downloadNow':
        downloadResults();
        sendResponse({ ok: true });
        break;

      case 'reset':
        resetSession();
        sendResponse({ ok: true });
        break;

      case 'pickConversation':
        scrapeSingleConversation(msg.conversation).then(sendResponse);
        return true;  // async response

      case 'pickCancelled':
        sendResponse({ ok: true });
        break;

      case 'discoveryProgress':
        // Relayed from content script — popup listens directly, nothing to do here
        sendResponse({ ok: true });
        break;

      case 'ping':
        sendResponse({ ok: true, timestamp: new Date().toISOString() });
        break;

      default:
        sendResponse({ ok: false, error: 'Unknown action: ' + msg.action });
    }
  } catch (e) {
    console.error('[Spider] Error handling message:', e);
    sendResponse({ ok: false, error: e.message });
  }
  return true;
});


// ═══════════════════════════════════════════════════════════════════════
//  BATCH SCRAPING
// ═══════════════════════════════════════════════════════════════════════

function startScraping(conversations, limit) {
  console.log(`[Spider] Starting scrape of ${conversations.length} conversations (limit: ${limit || 'all'})`);

  session = {
    ...DEFAULT_SESSION,
    state: 'running',
    total: conversations.length,
    allConversations: conversations,
    startTime: new Date().toISOString(),
    limit: limit || 0,
  };
  saveSession();
  startKeepAlive();
  processNext();
}

function pauseScraping() {
  if (session.state !== 'running') return;
  console.log('[Spider] Pausing...');
  session.state = 'paused';
  saveSession();
  // Current conversation will finish, then processNext() checks state and stops
}

function resumeScraping() {
  if (session.state !== 'paused') return;
  console.log(`[Spider] Resuming from index ${session.currentIndex}`);
  session.state = 'running';
  session.processingTab = null;
  saveSession();
  startKeepAlive();
  processNext();
}

function stopScraping(savePartial) {
  console.log(`[Spider] Stopping (savePartial=${savePartial})`);
  session.state = 'done';
  session.endTime = new Date().toISOString();

  // Clean up active timers
  activeTimers.forEach((data, tabId) => {
    if (data.timeout) clearTimeout(data.timeout);
    if (data.listener) chrome.tabs.onUpdated.removeListener(data.listener);
  });
  activeTimers.clear();

  // Close processing tab if any
  if (session.processingTab) {
    chrome.tabs.remove(session.processingTab, () => {
      if (chrome.runtime.lastError) {} // tab already gone
    });
    session.processingTab = null;
  }

  stopKeepAlive();
  saveSession();

  if (savePartial && session.conversations.length > 0) {
    downloadResults();
  }
}

function resetSession() {
  stopScraping(false);
  session = { ...DEFAULT_SESSION };
  chrome.storage.local.remove(['spiderSession']);
}


// ─── Process next conversation in queue ──────────────────────────────

function processNext() {
  // Check pause/stop
  if (session.state !== 'running') {
    console.log(`[Spider] processNext: state=${session.state}, halting`);
    if (session.state === 'paused') {
      // Save progress so we can resume
      saveSession();
    }
    return;
  }

  // Still processing a tab?
  if (session.processingTab !== null) {
    setTimeout(() => processNext(), 2000);
    return;
  }

  // Done?
  if (session.currentIndex >= session.total) {
    console.log('[Spider] All conversations processed');
    session.state = 'done';
    session.endTime = new Date().toISOString();
    saveSession();
    stopKeepAlive();

    if (session.conversations.length > 0) {
      downloadResults();
    }
    return;
  }

  const conversation = session.allConversations[session.currentIndex];
  if (!conversation) {
    session.currentIndex++;
    saveSession();
    processNext();
    return;
  }

  console.log(`[Spider] Processing ${session.currentIndex + 1}/${session.total}: ${conversation.title}`);

  // Open a background tab for this conversation
  chrome.tabs.create({ url: conversation.url, active: false }, (tab) => {
    if (chrome.runtime.lastError) {
      console.error('[Spider] Failed to create tab:', chrome.runtime.lastError.message);
      session.failed++;
      session.currentIndex++;
      saveSession();
      setTimeout(() => processNext(), 1000);
      return;
    }

    session.processingTab = tab.id;
    saveSession();

    const tabLoadListener = (tabId, changeInfo) => {
      if (tabId !== tab.id || changeInfo.status !== 'complete') return;

      clearTabTimer(tab.id);
      chrome.tabs.onUpdated.removeListener(tabLoadListener);

      console.log(`[Spider] Tab ${tab.id} loaded, waiting for content...`);

      // Wait for page render, then scroll to load content, then scrape
      setTimeout(() => {
        chrome.scripting.executeScript({
          target: { tabId: tab.id },
          func: () => {
            window.scrollTo(0, 0);
            setTimeout(() => window.scrollTo(0, document.body.scrollHeight), 800);
            return true;
          }
        }).then(() => {
          setTimeout(() => {
            chrome.tabs.sendMessage(tab.id, {
              action: 'scrapeThisPage',
              conversation,
            }, (response) => {
              if (chrome.runtime.lastError) {
                console.error(`[Spider] Scrape failed (messaging):`, chrome.runtime.lastError.message);
                session.failed++;
              } else if (response?.ok) {
                session.completed++;
                session.conversations.push(response.data);
                console.log(`[Spider] Scraped: "${conversation.title}" (${response.data.messageCount} msgs)`);
              } else {
                console.error(`[Spider] Scrape failed:`, response?.error);
                session.failed++;
              }

              cleanupAndNext(tab.id, tabLoadListener);
            });
          }, 2000);  // wait after scrolling
        }).catch(err => {
          console.error('[Spider] Script injection failed:', err);
          session.failed++;
          cleanupAndNext(tab.id, tabLoadListener);
        });
      }, 8000);  // wait for page render
    };

    chrome.tabs.onUpdated.addListener(tabLoadListener);

    // Timeout safety net (50 seconds)
    const timeout = setTimeout(() => {
      console.error(`[Spider] Timeout for: ${conversation.title}`);
      chrome.tabs.onUpdated.removeListener(tabLoadListener);
      session.failed++;
      cleanupAndNext(tab.id, tabLoadListener);
    }, 50000);

    activeTimers.set(tab.id, { timeout, listener: tabLoadListener });
  });
}


function cleanupAndNext(tabId, listener) {
  clearTabTimer(tabId);
  if (listener) chrome.tabs.onUpdated.removeListener(listener);

  chrome.tabs.remove(tabId, () => {
    if (chrome.runtime.lastError) {} // already gone

    session.processingTab = null;
    session.currentIndex++;
    saveSession();

    // Pace ourselves — 3 second gap between conversations
    setTimeout(() => processNext(), 3000);
  });
}


function clearTabTimer(tabId) {
  if (activeTimers.has(tabId)) {
    const data = activeTimers.get(tabId);
    if (data.timeout) clearTimeout(data.timeout);
    activeTimers.delete(tabId);
  }
}


// ═══════════════════════════════════════════════════════════════════════
//  SINGLE CONVERSATION SCRAPE (Pick Mode)
// ═══════════════════════════════════════════════════════════════════════

async function scrapeSingleConversation(conversation) {
  console.log(`[Spider] Single scrape: "${conversation.title}" → ${conversation.url}`);

  return new Promise((resolve) => {
    chrome.tabs.create({ url: conversation.url, active: false }, (tab) => {
      if (chrome.runtime.lastError) {
        resolve({ ok: false, error: chrome.runtime.lastError.message });
        return;
      }

      const listener = (tabId, changeInfo) => {
        if (tabId !== tab.id || changeInfo.status !== 'complete') return;
        chrome.tabs.onUpdated.removeListener(listener);

        // Wait for render + scroll + scrape
        setTimeout(() => {
          chrome.scripting.executeScript({
            target: { tabId: tab.id },
            func: () => {
              window.scrollTo(0, document.body.scrollHeight);
              return true;
            }
          }).then(() => {
            setTimeout(() => {
              chrome.tabs.sendMessage(tab.id, {
                action: 'scrapeThisPage',
                conversation,
              }, (response) => {
                chrome.tabs.remove(tab.id, () => {});

                if (chrome.runtime.lastError || !response?.ok) {
                  resolve({ ok: false, error: response?.error || chrome.runtime.lastError?.message || 'Unknown error' });
                  return;
                }

                // Download this single conversation immediately
                downloadSingleConversation(response.data);
                resolve({ ok: true, data: response.data });
              });
            }, 2000);
          }).catch(err => {
            chrome.tabs.remove(tab.id, () => {});
            resolve({ ok: false, error: err.message });
          });
        }, 8000);
      };

      chrome.tabs.onUpdated.addListener(listener);

      // Safety timeout
      setTimeout(() => {
        chrome.tabs.onUpdated.removeListener(listener);
        chrome.tabs.remove(tab.id, () => {});
        resolve({ ok: false, error: 'Timeout loading conversation' });
      }, 45000);
    });
  });
}


// ═══════════════════════════════════════════════════════════════════════
//  DOWNLOAD HELPERS
// ═══════════════════════════════════════════════════════════════════════

function downloadResults() {
  if (session.conversations.length === 0) {
    console.log('[Spider] No conversations to download');
    return;
  }

  const dataset = {
    metadata: {
      exportDate: new Date().toISOString(),
      totalConversations: session.total,
      successfullyScraped: session.completed,
      failed: session.failed,
      source: 'Anthill Spider v2.0',
      startTime: session.startTime,
      endTime: session.endTime || new Date().toISOString(),
      partial: session.completed < session.total,
    },
    conversations: session.conversations,
  };

  const json = JSON.stringify(dataset, null, 2);
  const blob = new Blob([json], { type: 'application/json' });
  const reader = new FileReader();

  reader.onloadend = () => {
    const partialTag = session.completed < session.total ? '_partial' : '';
    const filename = `chatgpt_conversations_${session.completed}of${session.total}${partialTag}.json`;

    chrome.downloads.download({
      url: reader.result,
      filename,
      saveAs: true,
    }, (downloadId) => {
      if (chrome.runtime.lastError) {
        console.error('[Spider] Download failed:', chrome.runtime.lastError);
      } else {
        console.log(`[Spider] Download started: ${filename} (${session.conversations.length} conversations)`);
      }
    });
  };

  reader.readAsDataURL(blob);
}


function downloadSingleConversation(data) {
  const json = JSON.stringify({
    metadata: {
      exportDate: new Date().toISOString(),
      totalConversations: 1,
      successfullyScraped: 1,
      failed: 0,
      source: 'Anthill Spider v2.0 (single export)',
    },
    conversations: [data],
  }, null, 2);

  const blob = new Blob([json], { type: 'application/json' });
  const reader = new FileReader();

  reader.onloadend = () => {
    // Sanitize title for filename
    const safeTitle = (data.title || 'conversation')
      .replace(/[^a-zA-Z0-9_\- ]/g, '')
      .replace(/\s+/g, '_')
      .substring(0, 60);

    chrome.downloads.download({
      url: reader.result,
      filename: `chatgpt_${safeTitle}.json`,
      saveAs: true,
    });
  };

  reader.readAsDataURL(blob);
}


// ═══════════════════════════════════════════════════════════════════════
//  CLEANUP
// ═══════════════════════════════════════════════════════════════════════

chrome.runtime.onSuspend.addListener(() => {
  console.log('[Spider] Service worker suspending');
  stopKeepAlive();

  // If we're running, mark as paused so we can auto-resume
  if (session.state === 'running') {
    session.state = 'paused';
    session.processingTab = null;
    saveSession();
  }

  activeTimers.forEach((data) => {
    if (data.timeout) clearTimeout(data.timeout);
  });
  activeTimers.clear();
});
