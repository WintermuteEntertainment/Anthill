// background.js — Anthill Collector Service Worker
// Orchestrates batch image scanning, auto-downloads images during scan.

console.log('[Collector] Background script loaded');

const DEFAULT_SESSION = {
  state: 'idle',          // idle | scanning | paused | done
  total: 0,
  completed: 0,
  failed: 0,
  allConversations: [],
  currentIndex: 0,
  processingTab: null,
  images: [],             // lightweight metadata (no data URLs stored here)
  downloadedCount: 0,
  startTime: null,
  autoDownload: true,     // download images as they're found
};

let session = { ...DEFAULT_SESSION };
// ── Separate storage for image data URLs (can be huge) ──────────────
// We use a Map in memory during the active scan session.
// Data URLs are only needed for gallery thumbnails while the popup is open.
const MAX_CACHE_SIZE = 100;
let imageDataCache = new Map();  // url-key → data URL

// Image type filter — which types to collect (empty = all)
let allowedTypes = [];  // e.g. ['dalle', 'generated', 'upload']

// Keep service worker alive during scanning via chrome.alarms
const KEEPALIVE_ALARM = 'collector-keepalive';

function startKeepAlive() {
  chrome.alarms.create(KEEPALIVE_ALARM, { periodInMinutes: 0.4 }); // ~24s
}
function stopKeepAlive() {
  chrome.alarms.clear(KEEPALIVE_ALARM);
}

chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name === KEEPALIVE_ALARM) {
    if (session.state !== 'scanning') {
      stopKeepAlive();
      return;
    }
    // Validate processing tab still exists
    if (session.processingTab !== null) {
      chrome.tabs.get(session.processingTab, (tab) => {
        if (chrome.runtime.lastError) {
          console.warn('[Collector] Keepalive: processing tab gone, clearing');
          session.processingTab = null;
          saveSession();
        }
      });
    }
    // Evict oldest cache entries if over limit
    if (imageDataCache.size > MAX_CACHE_SIZE) {
      const keys = imageDataCache.keys();
      while (imageDataCache.size > MAX_CACHE_SIZE) {
        imageDataCache.delete(keys.next().value);
      }
    }
  }
});

function saveSession() {
  // Save session metadata only — no data URLs, no huge blobs.
  // Images array contains only lightweight metadata objects.
  const saveable = {
    state: session.state,
    total: session.total,
    completed: session.completed,
    failed: session.failed,
    allConversations: session.allConversations,
    currentIndex: session.currentIndex,
    images: session.images.map(img => ({
      url: img.url,
      thumbUrl: img.thumbUrl || img.url,  // keep original URL for reference
      type: img.type,
      alt: img.alt,
      width: img.width,
      height: img.height,
      conversationTitle: img.conversationTitle,
      conversationUrl: img.conversationUrl,
      downloaded: img.downloaded || false,
      downloadPath: img.downloadPath || '',
    })),
    downloadedCount: session.downloadedCount,
    startTime: session.startTime,
    autoDownload: session.autoDownload,
  };
  chrome.storage.local.set({ collectorSession: saveable });
}

// Restore filters on startup
chrome.storage.local.get(['collectorFilters'], (r) => {
  if (r.collectorFilters) {
    const mapping = { dalle: ['dalle', 'generated'], upload: ['upload'], chart: ['chart'], web: ['image'] };
    allowedTypes = [];
    for (const [key, types] of Object.entries(mapping)) {
      if (r.collectorFilters[key] !== false) allowedTypes.push(...types);
    }
    // If all are enabled, clear the filter (no filtering needed)
    if (allowedTypes.length >= 5) allowedTypes = [];
    console.log('[Collector] Restored filters:', allowedTypes.length ? allowedTypes.join(', ') : 'all');
  }
});

// Restore session on startup
chrome.storage.local.get(['collectorSession'], (result) => {
  if (result.collectorSession) {
    session = { ...DEFAULT_SESSION, ...result.collectorSession };
    if (session.state === 'scanning') {
      session.state = 'paused';
      session.processingTab = null;
      saveSession();
    }
  }
});

// ═══════════════════════════════════════════════════════════════════════
//  MESSAGE ROUTER
// ═══════════════════════════════════════════════════════════════════════

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  try {
    switch (msg.action) {
      case 'scanCurrent':
        sendResponse({ ok: true });
        break;

      case 'startBatchScan':
        startBatchScan(msg.conversations);
        sendResponse({ ok: true });
        break;

      case 'getStatus':
        sendResponse({
          state: session.state,
          total: session.total,
          completed: session.completed,
          failed: session.failed,
          imageCount: session.images.length,
          downloadedCount: session.downloadedCount,
          startTime: session.startTime,
        });
        break;

      case 'getImages':
        // Return images with cached data URLs for gallery thumbnails
        const imagesWithData = session.images.map(img => {
          const cached = imageDataCache.get(img.url);
          return cached ? { ...img, dataUrl: cached } : img;
        });
        sendResponse({ ok: true, images: imagesWithData });
        break;

      case 'addImages':
        addImages(msg.images);
        sendResponse({ ok: true, total: session.images.length });
        break;

      case 'setFilters':
        allowedTypes = msg.allowedTypes || [];
        console.log('[Collector] Filter set:', allowedTypes.length ? allowedTypes.join(', ') : 'all');
        sendResponse({ ok: true });
        break;

      case 'pause':
        pauseScan();
        sendResponse({ ok: true });
        break;

      case 'resume':
        resumeScan();
        sendResponse({ ok: true });
        break;

      case 'stop':
        stopScan();
        sendResponse({ ok: true });
        break;

      case 'reset':
        resetSession();
        sendResponse({ ok: true });
        break;

      case 'downloadImage':
        downloadSingleImage(msg.image, msg.index);
        sendResponse({ ok: true });
        break;

      case 'downloadAll':
        downloadAllImages(msg.images);
        sendResponse({ ok: true });
        break;

      case 'discoveryProgress':
        sendResponse({ ok: true });
        break;

      case 'ping':
        sendResponse({ ok: true });
        break;

      default:
        sendResponse({ ok: false, error: 'Unknown action: ' + msg.action });
    }
  } catch (e) {
    sendResponse({ ok: false, error: e.message });
  }
  return true;
});


// ═══════════════════════════════════════════════════════════════════════
//  IMAGE MANAGEMENT
// ═══════════════════════════════════════════════════════════════════════

function addImages(newImages) {
  const existingUrls = new Set(session.images.map(i => i.url));
  let added = 0;
  let filtered = 0;

  for (const img of newImages) {
    // Use the original URL as the dedup key (not the data URL)
    const key = img.originalUrl || img.url;
    if (existingUrls.has(key)) continue;

    // Apply type filter if set
    if (allowedTypes.length > 0) {
      const imgType = img.type || 'image';
      if (!allowedTypes.includes(imgType)) {
        filtered++;
        continue;
      }
    }

    // Cache data URL in memory for gallery thumbnails (with LRU eviction)
    if (img.dataUrl) {
      imageDataCache.set(key, img.dataUrl);
      if (imageDataCache.size > MAX_CACHE_SIZE) {
        const oldest = imageDataCache.keys().next().value;
        imageDataCache.delete(oldest);
      }
    }

    // Store lightweight metadata only
    const meta = {
      url: key,
      thumbUrl: img.dataUrl ? key : img.url,
      type: img.type || 'image',
      alt: img.alt || '',
      width: img.width || 0,
      height: img.height || 0,
      conversationTitle: img.conversationTitle || '',
      conversationUrl: img.conversationUrl || '',
      downloaded: false,
      downloadPath: '',
    };

    session.images.push(meta);
    existingUrls.add(key);
    added++;

    // Auto-download immediately if enabled
    if (session.autoDownload) {
      const idx = session.images.length - 1;
      const path = buildFilename(meta, idx);
      meta.downloadPath = path;
      meta.downloaded = true;
      session.downloadedCount++;

      // Use downloadSingleImage which handles data URL → objectURL conversion
      downloadSingleImage({ ...meta, dataUrl: img.dataUrl }, idx);
    }
  }

  console.log(`[Collector] Added ${added} new images (${session.images.length} total, ${session.downloadedCount} downloaded${filtered ? `, ${filtered} filtered out` : ''})`);
  saveSession();
}


// ═══════════════════════════════════════════════════════════════════════
//  BATCH SCANNING
// ═══════════════════════════════════════════════════════════════════════

function startBatchScan(conversations) {
  session = {
    ...DEFAULT_SESSION,
    state: 'scanning',
    total: conversations.length,
    allConversations: conversations,
    startTime: new Date().toISOString(),
    images: [...session.images],
    downloadedCount: session.downloadedCount,
  };
  imageDataCache.clear();
  saveSession();
  startKeepAlive();
  processNext();
}

function pauseScan() {
  if (session.state === 'scanning') {
    session.state = 'paused';
    saveSession();
  }
}

function resumeScan() {
  if (session.state === 'paused') {
    session.state = 'scanning';
    session.processingTab = null;
    saveSession();
    startKeepAlive();
    processNext();
  }
}

function stopScan() {
  session.state = 'done';
  cleanupProcessingTab();
  stopKeepAlive();
  saveSession();
}

function resetSession() {
  cleanupProcessingTab();
  stopKeepAlive();
  session = { ...DEFAULT_SESSION };
  imageDataCache.clear();
  chrome.storage.local.remove(['collectorSession']);
}

function cleanupProcessingTab() {
  if (session.processingTab) {
    chrome.tabs.remove(session.processingTab, () => {
      if (chrome.runtime.lastError) {}
    });
    session.processingTab = null;
  }
}

function processNext() {
  if (session.state !== 'scanning') return;
  if (session.processingTab !== null) {
    setTimeout(processNext, 2000);
    return;
  }
  if (session.currentIndex >= session.total) {
    session.state = 'done';
    saveSession();
    stopKeepAlive();

    // Notify popup that scan is complete
    chrome.runtime.sendMessage({
      action: 'batchComplete',
      total: session.images.length,
      downloaded: session.downloadedCount,
    }).catch(() => {});

    console.log(`[Collector] ✅ Batch scan complete: ${session.images.length} images found, ${session.downloadedCount} downloaded`);
    return;
  }

  const conversation = session.allConversations[session.currentIndex];
  if (!conversation) {
    session.currentIndex++;
    saveSession();
    processNext();
    return;
  }

  console.log(`[Collector] Scanning ${session.currentIndex + 1}/${session.total}: ${conversation.title}`);

  // Notify popup of progress
  chrome.runtime.sendMessage({
    action: 'batchProgress',
    index: session.currentIndex,
    total: session.total,
    title: conversation.title,
    imagesSoFar: session.images.length,
    downloadedSoFar: session.downloadedCount,
  }).catch(() => {});

  chrome.tabs.create({ url: conversation.url, active: false }, (tab) => {
    if (chrome.runtime.lastError) {
      session.failed++;
      session.currentIndex++;
      saveSession();
      setTimeout(processNext, 1000);
      return;
    }

    session.processingTab = tab.id;
    saveSession();

    let processed = false;  // Guard against race between listener and timeout

    const listener = (tabId, changeInfo) => {
      if (tabId !== tab.id || changeInfo.status !== 'complete') return;
      chrome.tabs.onUpdated.removeListener(listener);

      // Wait for render, scroll, then scan images
      setTimeout(() => {
        if (processed) return;
        chrome.tabs.sendMessage(tab.id, { action: 'scanImages' }, (response) => {
          if (processed) return;
          processed = true;

          if (chrome.runtime.lastError || !response?.ok) {
            console.error(`[Collector] Failed to scan: ${conversation.title}`);
            session.failed++;
          } else {
            const imgs = response.images || [];
            if (imgs.length > 0) {
              addImages(imgs);  // auto-downloads happen inside addImages()
              console.log(`[Collector] "${conversation.title}": ${imgs.length} images`);
            }
            session.completed++;
          }

          // Close tab and continue
          chrome.tabs.remove(tab.id, () => {
            if (chrome.runtime.lastError) {}
            session.processingTab = null;
            session.currentIndex++;
            saveSession();
            setTimeout(processNext, 2000);
          });
        });
      }, 10000);  // 10s for page load + scroll
    };

    chrome.tabs.onUpdated.addListener(listener);

    // Safety timeout
    setTimeout(() => {
      chrome.tabs.onUpdated.removeListener(listener);
      if (!processed && session.processingTab === tab.id) {
        processed = true;
        session.failed++;
        chrome.tabs.remove(tab.id, () => {
          if (chrome.runtime.lastError) {}
          session.processingTab = null;
          session.currentIndex++;
          saveSession();
          setTimeout(processNext, 2000);
        });
      }
    }, 50000);
  });
}


// ═══════════════════════════════════════════════════════════════════════
//  DOWNLOADS
// ═══════════════════════════════════════════════════════════════════════

function downloadSingleImage(image, index) {
  const filename = buildFilename(image, index);
  const src = image.dataUrl || image.url;

  if (src.startsWith('data:')) {
    // Convert data URL to blob + object URL (more memory efficient)
    fetch(src)
      .then(res => res.blob())
      .then(blob => {
        const objectUrl = URL.createObjectURL(blob);
        chrome.downloads.download({
          url: objectUrl,
          filename,
          saveAs: false,
          conflictAction: 'uniquify',
        }, () => {
          setTimeout(() => URL.revokeObjectURL(objectUrl), 60000);
          if (chrome.runtime.lastError) {
            console.warn(`[Collector] Download failed: ${chrome.runtime.lastError.message}`);
          }
        });
      })
      .catch(err => console.warn(`[Collector] Data URL conversion failed: ${err.message}`));
  } else {
    chrome.downloads.download({
      url: src,
      filename,
      saveAs: false,
      conflictAction: 'uniquify',
    }, () => {
      if (chrome.runtime.lastError) {
        console.warn(`[Collector] Download failed: ${chrome.runtime.lastError.message}`);
      }
    });
  }
}

function downloadAllImages(images) {
  console.log(`[Collector] Downloading ${images.length} images...`);

  // Download with a staggered delay to avoid overwhelming the browser
  images.forEach((img, i) => {
    setTimeout(() => {
      downloadSingleImage(img, i);
    }, i * 300);  // 300ms between each download
  });
}

function buildFilename(image, index) {
  // Sanitize conversation title for filename
  const title = (image.conversationTitle || 'chatgpt')
    .replace(/[^a-zA-Z0-9_\- ]/g, '')
    .replace(/\s+/g, '_')
    .substring(0, 50);

  const typeLabel = image.type || 'image';
  const idx = String(index + 1).padStart(4, '0');

  // Determine extension from URL or type
  let ext = 'png';
  const url = image.dataUrl || image.url || '';
  if (url.startsWith('data:image/jpeg') || url.startsWith('data:image/jpg')) ext = 'jpg';
  else if (url.startsWith('data:image/svg')) ext = 'svg';
  else if (url.startsWith('data:image/gif')) ext = 'gif';
  else if (url.startsWith('data:image/webp')) ext = 'webp';
  else if (!url.startsWith('data:')) {
    const match = url.match(/\.(png|jpg|jpeg|gif|webp|svg|bmp)(\?|$)/i);
    if (match) ext = match[1].toLowerCase();
    if (ext === 'jpeg') ext = 'jpg';
  }

  return `AnthillCollector/${title}/${typeLabel}_${idx}.${ext}`;
}


// ═══════════════════════════════════════════════════════════════════════
//  CLEANUP
// ═══════════════════════════════════════════════════════════════════════

chrome.runtime.onSuspend.addListener(() => {
  stopKeepAlive();
  if (session.state === 'scanning') {
    session.state = 'paused';
    session.processingTab = null;
    saveSession();
  }
});
