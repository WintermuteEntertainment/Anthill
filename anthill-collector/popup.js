// popup.js — Anthill Collector UI
// Image gallery with thumbnail grid, filters, lightbox, batch scanning

let allImages = [];
let selectedUrls = new Set();
let activeFilter = 'all';
let statusInterval = null;

document.addEventListener('DOMContentLoaded', () => {
  // Controls
  document.getElementById('scanPageBtn').addEventListener('click', scanCurrentPage);
  document.getElementById('findAllBtn').addEventListener('click', findAllConversations);
  document.getElementById('batchBtn').addEventListener('click', startBatchScan);

  // Progress
  document.getElementById('pauseBtn').addEventListener('click', () => sendAction('pause'));
  document.getElementById('resumeScanBtn').addEventListener('click', () => {
    sendAction('resume');
    setTimeout(refreshStatus, 300);
  });
  document.getElementById('stopBtn').addEventListener('click', () => {
    if (confirm('Stop scanning? Images found so far will be kept.')) {
      sendAction('stop');
      stopPolling();
      setTimeout(refreshStatus, 500);
    }
  });

  // Gallery actions
  document.getElementById('selectAllBtn').addEventListener('click', toggleSelectAll);
  document.getElementById('downloadSelBtn').addEventListener('click', downloadSelected);

  // Lightbox
  document.getElementById('lightbox').addEventListener('click', (e) => {
    if (e.target === e.currentTarget || e.target.id === 'lbClose') closeLightbox();
  });
  document.getElementById('lbClose').addEventListener('click', closeLightbox);

  // Resume banner
  document.getElementById('resumeBtn').addEventListener('click', () => {
    sendAction('resume');
    setTimeout(refreshStatus, 300);
  });
  document.getElementById('discardBtn').addEventListener('click', () => {
    sendAction('reset');
    allImages = [];
    renderGallery();
    setTimeout(refreshStatus, 300);
  });

  // Listen for batch progress
  chrome.runtime.onMessage.addListener((msg) => {
    if (msg.action === 'batchProgress') {
      document.getElementById('statusText').textContent =
        `${msg.index + 1}/${msg.total}: ${msg.title} (${msg.imagesSoFar} images, ${msg.downloadedSoFar || 0} saved)`;
    }
    if (msg.action === 'batchComplete') {
      refreshStatus();
    }
    if (msg.action === 'discoveryProgress') {
      const el = document.getElementById('discoveryScanStatus');
      el.classList.remove('hidden');
      el.textContent = '';
      const spinner = document.createElement('span');
      spinner.className = 'spinner';
      el.appendChild(spinner);
      el.appendChild(document.createTextNode(' ' + msg.message));
    }
  });

  // Load saved limit
  chrome.storage.local.get(['collectorLimit'], (r) => {
    if (r.collectorLimit) document.getElementById('limitInput').value = r.collectorLimit;
  });
  document.getElementById('limitInput').addEventListener('change', (e) => {
    chrome.storage.local.set({ collectorLimit: parseInt(e.target.value) || 0 });
  });

  // ─── Image type filter checkboxes ────────────────────────────────
  const filterBoxes = {
    dalle:   document.getElementById('filterDalle'),
    upload:  document.getElementById('filterUpload'),
    chart:   document.getElementById('filterChart'),
    web:     document.getElementById('filterWeb'),
  };

  // Load saved filter prefs
  chrome.storage.local.get(['collectorFilters'], (r) => {
    if (r.collectorFilters) {
      for (const [key, el] of Object.entries(filterBoxes)) {
        if (typeof r.collectorFilters[key] === 'boolean') {
          el.checked = r.collectorFilters[key];
        }
      }
    }
    // Push current filters to background on load
    syncFiltersToBackground();
  });

  // Save and sync on any change
  for (const el of Object.values(filterBoxes)) {
    el.addEventListener('change', () => {
      const prefs = {};
      for (const [key, box] of Object.entries(filterBoxes)) {
        prefs[key] = box.checked;
      }
      chrome.storage.local.set({ collectorFilters: prefs });
      syncFiltersToBackground();
    });
  }

  // Initial state
  refreshStatus();
});


// ═══════════════════════════════════════════════════════════════════════
//  STATE
// ═══════════════════════════════════════════════════════════════════════

function refreshStatus() {
  chrome.runtime.sendMessage({ action: 'getStatus' }, (status) => {
    if (chrome.runtime.lastError || !status) return;

    // Always load stored images
    chrome.runtime.sendMessage({ action: 'getImages' }, (resp) => {
      if (resp?.ok && resp.images?.length) {
        allImages = resp.images;
        renderGallery();
      }
    });

    switch (status.state) {
      case 'scanning':
        showProgress(true);
        document.getElementById('pauseBtn').classList.remove('hidden');
        document.getElementById('resumeScanBtn').classList.add('hidden');
        document.getElementById('scanLabel').textContent = 'Scanning conversations...';
        updateProgress(status);
        startPolling();
        break;

      case 'paused':
        if (!statusInterval) {
          // Popup just opened — show resume banner
          document.getElementById('resumeBanner').classList.remove('hidden');
          document.getElementById('resumeInfo').textContent =
            `${status.completed}/${status.total} conversations scanned, ${status.imageCount} images found.`;
          showProgress(false);
        } else {
          showProgress(true);
          document.getElementById('pauseBtn').classList.add('hidden');
          document.getElementById('resumeScanBtn').classList.remove('hidden');
          document.getElementById('scanLabel').textContent = 'Paused';
          updateProgress(status);
        }
        break;

      case 'done':
        showProgress(false);
        stopPolling();
        break;

      default:
        showProgress(false);
        stopPolling();
        break;
    }
  });
}

function showProgress(show) {
  document.getElementById('progressPanel').classList.toggle('hidden', !show);
  document.getElementById('galleryPanel').classList.toggle('hidden', allImages.length === 0);
}

function updateProgress(status) {
  const pct = status.total > 0 ? ((status.completed + status.failed) / status.total * 100) : 0;
  document.getElementById('progressFill').style.width = pct + '%';
  document.getElementById('statsDone').textContent = `Scanned: ${status.completed}`;
  document.getElementById('statsImages').textContent = `Images: ${status.imageCount}`;
  document.getElementById('statsFailed').textContent = `Failed: ${status.failed}`;
  if (status.downloadedCount > 0) {
    document.getElementById('statsFailed').textContent += ` | Saved: ${status.downloadedCount}`;
  }
}

function startPolling() {
  if (statusInterval) return;
  statusInterval = setInterval(refreshStatus, 2000);
}

function stopPolling() {
  if (statusInterval) { clearInterval(statusInterval); statusInterval = null; }
}


// ═══════════════════════════════════════════════════════════════════════
//  SCAN CURRENT PAGE
// ═══════════════════════════════════════════════════════════════════════

async function scanCurrentPage() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.url || (!tab.url.includes('chatgpt.com') && !tab.url.includes('chat.openai.com'))) {
    alert('Please open a ChatGPT conversation first.');
    return;
  }

  const btn = document.getElementById('scanPageBtn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Scanning...';

  try {
    await ensureContentScript(tab.id);
    const response = await tabMessage(tab.id, { action: 'scanImages' });

    if (!response?.ok) {
      throw new Error(response?.error || 'Failed to scan page');
    }

    const found = response.images || [];
    if (found.length === 0) {
      alert('No images found in this conversation.');
    } else {
      // Store in background
      chrome.runtime.sendMessage({ action: 'addImages', images: found });

      // Merge into local list
      const existingUrls = new Set(allImages.map(i => i.url));
      for (const img of found) {
        if (!existingUrls.has(img.url)) {
          allImages.push(img);
          existingUrls.add(img.url);
        }
      }
      renderGallery();
    }
  } catch (e) {
    alert('Error: ' + e.message);
  }

  btn.disabled = false;
  btn.innerHTML = '&#x1F50D; Scan This Conversation';
}


// ═══════════════════════════════════════════════════════════════════════
//  FIND ALL / BATCH SCAN
// ═══════════════════════════════════════════════════════════════════════

async function findAllConversations() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!isChatGPT(tab)) { alert('Please open ChatGPT first.'); return; }

  const btn = document.getElementById('findAllBtn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Scanning...';

  const statusEl = document.getElementById('discoveryScanStatus');
  statusEl.classList.remove('hidden');
  statusEl.textContent = 'Starting sidebar scan...';

  try {
    await ensureContentScript(tab.id);
    const resp = await tabMessage(tab.id, { action: 'findAllConversations' });

    if (!resp?.ok) throw new Error(resp?.error || 'Failed');

    statusEl.innerHTML = `&#10003; Found <strong>${resp.total}</strong> conversations. Set a limit or click Batch Scan.`;
  } catch (e) {
    statusEl.textContent = `Error: ${e.message}`;
  }

  btn.disabled = false;
  btn.innerHTML = '&#x1F50D; Find All';
}

async function startBatchScan() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!isChatGPT(tab)) { alert('Please open ChatGPT first.'); return; }

  const limit = parseInt(document.getElementById('limitInput').value) || 0;

  const btn = document.getElementById('batchBtn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Loading sidebar...';

  try {
    await ensureContentScript(tab.id);
    const resp = await tabMessage(tab.id, { action: 'extractLinks', limit });

    if (!resp?.ok || !resp.conversations?.length) {
      throw new Error(resp?.error || 'No conversations found.');
    }

    const conversations = resp.conversations;
    const limitNote = limit > 0 ? ` (last ${limit} of ${resp.total})` : '';

    if (!confirm(`Scan ${conversations.length} conversations${limitNote} for images?\n\nTabs open in the background. ~15 seconds per conversation.`)) {
      btn.disabled = false;
      btn.innerHTML = '&#x1F680; Batch Scan';
      return;
    }

    chrome.runtime.sendMessage({
      action: 'startBatchScan',
      conversations,
    });

    setTimeout(refreshStatus, 500);
  } catch (e) {
    alert('Error: ' + e.message);
  }

  btn.disabled = false;
  btn.innerHTML = '&#x1F680; Batch Scan';
}


// ═══════════════════════════════════════════════════════════════════════
//  IMAGE GALLERY
// ═══════════════════════════════════════════════════════════════════════

function renderGallery() {
  const panel = document.getElementById('galleryPanel');
  panel.classList.remove('hidden');

  // Count by type
  const typeCounts = {};
  for (const img of allImages) {
    typeCounts[img.type] = (typeCounts[img.type] || 0) + 1;
  }

  // Render filter chips
  const filtersEl = document.getElementById('galleryFilters');
  let filtersHtml = `<span class="filter-chip ${activeFilter === 'all' ? 'active' : ''}" data-filter="all">All<span class="chip-count">${allImages.length}</span></span>`;
  const typeLabels = { dalle: 'DALL-E', upload: 'Uploads', chart: 'Charts', generated: 'Generated', image: 'Other' };
  for (const [type, count] of Object.entries(typeCounts).sort((a, b) => b[1] - a[1])) {
    const label = typeLabels[type] || type;
    filtersHtml += `<span class="filter-chip ${activeFilter === type ? 'active' : ''}" data-filter="${type}">${label}<span class="chip-count">${count}</span></span>`;
  }
  filtersEl.innerHTML = filtersHtml;

  // Wire filter clicks
  filtersEl.querySelectorAll('.filter-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      activeFilter = chip.dataset.filter;
      renderGallery();
    });
  });

  // Filter images
  const filtered = activeFilter === 'all' ? allImages : allImages.filter(i => i.type === activeFilter);

  // Update count
  document.getElementById('galleryCount').textContent =
    `${filtered.length} image${filtered.length !== 1 ? 's' : ''}${activeFilter !== 'all' ? ` (${activeFilter})` : ''}`;

  // Render grid
  const grid = document.getElementById('galleryGrid');
  if (filtered.length === 0) {
    grid.innerHTML = '<div class="empty-gallery">No images found yet</div>';
    updateDownloadBtn();
    return;
  }

  grid.innerHTML = filtered.map((img, i) => {
    const sel = selectedUrls.has(img.url) ? 'selected' : '';
    const badgeClass = `badge-${img.type}`;
    const typeLabel = img.type || 'image';
    const convLabel = img.conversationTitle || '';
    const dlBadge = img.downloaded ? '✅' : '';
    // Prefer captured data URL over remote URL (which expires)
    const thumbSrc = img.dataUrl || img.url;

    return `
      <div class="thumb ${sel}" data-url="${esc(img.url)}" data-index="${i}">
        <img src="${esc(thumbSrc)}" loading="lazy" alt="${esc(img.alt)}" onerror="this.style.display='none'">
        <span class="badge ${badgeClass}">${typeLabel}</span>
        <span class="check">&#10003;</span>
        ${dlBadge ? `<span class="dl-badge">${dlBadge}</span>` : ''}
        <span class="conv-label">${esc(convLabel)}</span>
      </div>`;
  }).join('');

  // Wire click events: click = select, double-click = lightbox
  grid.querySelectorAll('.thumb').forEach(thumb => {
    thumb.addEventListener('click', (e) => {
      e.preventDefault();
      toggleSelect(thumb);
    });

    thumb.addEventListener('dblclick', (e) => {
      e.preventDefault();
      const idx = parseInt(thumb.dataset.index);
      openLightbox(filtered[idx]);
    });

    // Right-click to toggle selection
    thumb.addEventListener('contextmenu', (e) => {
      e.preventDefault();
      toggleSelect(thumb);
    });
  });

  updateDownloadBtn();
}

function toggleSelect(thumb) {
  const url = thumb.dataset.url;
  if (selectedUrls.has(url)) {
    selectedUrls.delete(url);
    thumb.classList.remove('selected');
  } else {
    selectedUrls.add(url);
    thumb.classList.add('selected');
  }
  updateDownloadBtn();
}

function toggleSelectAll() {
  const filtered = activeFilter === 'all' ? allImages : allImages.filter(i => i.type === activeFilter);
  const allSelected = filtered.every(i => selectedUrls.has(i.url));

  if (allSelected) {
    filtered.forEach(i => selectedUrls.delete(i.url));
  } else {
    filtered.forEach(i => selectedUrls.add(i.url));
  }
  renderGallery();
}

function updateDownloadBtn() {
  const btn = document.getElementById('downloadSelBtn');
  const count = selectedUrls.size;
  btn.disabled = count === 0;
  btn.textContent = count > 0 ? `\u2B07 Download (${count})` : '\u2B07 Download';

  const selectBtn = document.getElementById('selectAllBtn');
  const filtered = activeFilter === 'all' ? allImages : allImages.filter(i => i.type === activeFilter);
  const allSel = filtered.length > 0 && filtered.every(i => selectedUrls.has(i.url));
  selectBtn.textContent = allSel ? 'Deselect All' : 'Select All';
}


// ═══════════════════════════════════════════════════════════════════════
//  DOWNLOAD
// ═══════════════════════════════════════════════════════════════════════

async function downloadSelected() {
  const toDownload = allImages.filter(i => selectedUrls.has(i.url));
  if (toDownload.length === 0) return;

  const btn = document.getElementById('downloadSelBtn');
  btn.disabled = true;
  btn.textContent = `⏳ Zipping ${toDownload.length} images...`;

  try {
    const zip = new JSZip();
    let completed = 0;

    for (const img of toDownload) {
      const src = img.dataUrl || img.url;
      const folder = sanitizeFilename(img.conversationTitle || 'chatgpt');
      const typeLabel = img.type || 'image';
      const idx = String(completed + 1).padStart(4, '0');
      let ext = guessExtension(src);
      const filename = `${folder}/${typeLabel}_${idx}.${ext}`;

      try {
        let blob;
        if (src.startsWith('data:')) {
          blob = dataUrlToBlob(src);
          if (!blob) { completed++; continue; }
        } else {
          // Remote URL — fetch it (may fail if token expired)
          const resp = await fetch(src);
          if (!resp.ok) { completed++; continue; }
          const ct = resp.headers.get('content-type') || '';
          if (ct.includes('json') || ct.includes('html')) { completed++; continue; }
          blob = await resp.blob();
        }
        zip.file(filename, blob);
      } catch (e) {
        console.warn(`[Collector] Skipping ${filename}:`, e.message);
      }

      completed++;
      if (completed % 25 === 0 || completed === toDownload.length) {
        btn.textContent = `⏳ Zipping... ${completed}/${toDownload.length}`;
      }
    }

    btn.textContent = '⏳ Generating ZIP...';

    const zipBlob = await zip.generateAsync({
      type: 'blob',
      compression: 'STORE',   // images are already compressed, no point re-compressing
    }, (meta) => {
      if (meta.percent) {
        btn.textContent = `⏳ ZIP ${Math.round(meta.percent)}%...`;
      }
    });

    // Trigger download of the single zip file
    const url = URL.createObjectURL(zipBlob);
    const a = document.createElement('a');
    a.href = url;
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').substring(0, 19);
    a.download = `AnthillCollector_${timestamp}.zip`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 60000);

    btn.textContent = `✅ Done! ${completed} images zipped`;
    btn.disabled = false;
    setTimeout(() => updateDownloadBtn(), 3000);

  } catch (e) {
    console.error('[Collector] Zip failed:', e);
    alert('Zip failed: ' + e.message);
    btn.textContent = '⬇ Download';
    btn.disabled = false;
  }
}

function sanitizeFilename(name) {
  return (name || 'chatgpt')
    .replace(/[^a-zA-Z0-9_\- ]/g, '')
    .replace(/\s+/g, '_')
    .substring(0, 50);
}

function guessExtension(url) {
  if (url.startsWith('data:image/jpeg') || url.startsWith('data:image/jpg')) return 'jpg';
  if (url.startsWith('data:image/svg')) return 'svg';
  if (url.startsWith('data:image/gif')) return 'gif';
  if (url.startsWith('data:image/webp')) return 'webp';
  if (!url.startsWith('data:')) {
    const match = url.match(/\.(png|jpg|jpeg|gif|webp|svg|bmp)(\?|$)/i);
    if (match) { let e = match[1].toLowerCase(); return e === 'jpeg' ? 'jpg' : e; }
  }
  return 'png';
}

function dataUrlToBlob(dataUrl) {
  try {
    const [header, data] = dataUrl.split(',');
    const mime = header.match(/:(.*?);/)?.[1] || 'image/png';
    const isBase64 = header.indexOf('base64') !== -1;
    const decoded = isBase64 ? atob(data) : decodeURIComponent(data);
    const arr = new Uint8Array(decoded.length);
    for (let i = 0; i < decoded.length; i++) arr[i] = decoded.charCodeAt(i);
    return new Blob([arr], { type: mime });
  } catch (e) {
    console.warn('[Collector] dataUrlToBlob failed:', e.message);
    return null;
  }
}


// ═══════════════════════════════════════════════════════════════════════
//  LIGHTBOX
// ═══════════════════════════════════════════════════════════════════════

function openLightbox(img) {
  const lb = document.getElementById('lightbox');
  document.getElementById('lbImg').src = img.dataUrl || img.url;

  const parts = [];
  if (img.type) parts.push(img.type.toUpperCase());
  if (img.width && img.height) parts.push(`${img.width} x ${img.height}`);
  if (img.conversationTitle) parts.push(img.conversationTitle);
  if (img.alt && img.alt.length < 100) parts.push(img.alt);
  document.getElementById('lbInfo').textContent = parts.join(' — ');

  lb.classList.remove('hidden');
}

function closeLightbox() {
  document.getElementById('lightbox').classList.add('hidden');
  document.getElementById('lbImg').src = '';
}


// ═══════════════════════════════════════════════════════════════════════
//  HELPERS
// ═══════════════════════════════════════════════════════════════════════

function isChatGPT(tab) {
  return tab?.url && (tab.url.includes('chatgpt.com') || tab.url.includes('chat.openai.com'));
}

function sendAction(action) {
  chrome.runtime.sendMessage({ action }, () => {
    if (chrome.runtime.lastError) console.error(chrome.runtime.lastError);
  });
}

function tabMessage(tabId, message) {
  return new Promise((resolve, reject) => {
    chrome.tabs.sendMessage(tabId, message, (response) => {
      if (chrome.runtime.lastError) reject(new Error(chrome.runtime.lastError.message));
      else resolve(response);
    });
  });
}

async function ensureContentScript(tabId) {
  try {
    const resp = await tabMessage(tabId, { action: 'ping' });
    if (resp?.ok) return;
  } catch {}
  await chrome.scripting.executeScript({ target: { tabId }, files: ['content.js'] });
  await sleep(500);
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function getActiveFilterTypes() {
  // Map checkbox IDs to the image type values used by classifyImage()
  const mapping = {
    dalle:  ['dalle', 'generated'],  // both AI-generated types
    upload: ['upload'],
    chart:  ['chart'],
    web:    ['image'],               // generic/web images
  };
  const allowed = [];
  for (const [key, types] of Object.entries(mapping)) {
    const el = document.getElementById('filter' + key.charAt(0).toUpperCase() + key.slice(1));
    if (el && el.checked) allowed.push(...types);
  }
  return allowed;
}

function syncFiltersToBackground() {
  const allowedTypes = getActiveFilterTypes();
  chrome.runtime.sendMessage({ action: 'setFilters', allowedTypes }, () => {
    if (chrome.runtime.lastError) console.warn('syncFilters:', chrome.runtime.lastError);
  });
}

function esc(s) {
  if (!s) return '';
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')
    .replace(/`/g, '&#96;');
}
