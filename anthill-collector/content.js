// content.js — Anthill Collector
// Runs on ChatGPT pages. Extracts images and sidebar links.

console.log('[Collector] Content script loaded on', location.href);

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  switch (msg.action) {
    case 'scanImages':
      scanPageImages().then(sendResponse);
      return true;

    case 'extractLinks':
      extractSidebarLinks(msg.limit || 0).then(sendResponse);
      return true;

    case 'findAllConversations':
      findAllConversations().then(sendResponse);
      return true;

    case 'ping':
      sendResponse({ ok: true });
      return false;
  }
});


// ═══════════════════════════════════════════════════════════════════════
//  IMAGE SCANNING
// ═══════════════════════════════════════════════════════════════════════

async function scanPageImages() {
  try {
    // Scroll to load all lazy-loaded content
    await scrollToLoadAll();

    const images = [];
    const seenUrls = new Set();
    const conversationTitle = getConversationTitle();

    // ─── 1. Standard <img> tags ──────────────────────────────────────
    for (const img of document.querySelectorAll('img')) {
      const src = img.src || img.getAttribute('data-src') || '';
      if (!src || seenUrls.has(src)) continue;
      if (isUIImage(src, img)) continue;

      seenUrls.add(src);
      images.push({
        url: src,
        type: classifyImage(src, img),
        alt: img.alt || '',
        width: img.naturalWidth || img.width || 0,
        height: img.naturalHeight || img.height || 0,
        conversationTitle,
        conversationUrl: location.href,
      });
    }

    // ─── 2. Background images in style attributes ────────────────────
    for (const el of document.querySelectorAll('[style*="background-image"]')) {
      const match = el.style.backgroundImage.match(/url\(["']?(.*?)["']?\)/);
      if (!match) continue;
      const url = match[1];
      if (!url || seenUrls.has(url) || isUIUrl(url)) continue;

      seenUrls.add(url);
      images.push({
        url,
        type: classifyImage(url, el),
        alt: '',
        width: el.offsetWidth || 0,
        height: el.offsetHeight || 0,
        conversationTitle,
        conversationUrl: location.href,
      });
    }

    // ─── 3. <canvas> elements (charts/code interpreter output) ───────
    const canvases = document.querySelectorAll('canvas');
    for (let i = 0; i < canvases.length; i++) {
      const canvas = canvases[i];
      if (canvas.width < 50 || canvas.height < 50) continue; // skip tiny UI canvases

      try {
        const dataUrl = canvas.toDataURL('image/png');
        if (dataUrl === 'data:,') continue; // tainted or empty canvas

        const canvasId = `canvas_${i}_${canvas.width}x${canvas.height}`;
        if (seenUrls.has(canvasId)) continue;
        seenUrls.add(canvasId);

        images.push({
          url: dataUrl,
          type: 'chart',
          alt: `Chart/plot (${canvas.width}x${canvas.height})`,
          width: canvas.width,
          height: canvas.height,
          conversationTitle,
          conversationUrl: location.href,
          isDataUrl: true,
        });
      } catch {
        // Canvas tainted by cross-origin data — can't extract
      }
    }

    // ─── 4. SVG images that might be inline charts ───────────────────
    for (const svg of document.querySelectorAll('svg')) {
      // Only grab large SVGs that look like charts, not UI icons
      if (svg.clientWidth < 200 || svg.clientHeight < 100) continue;
      // Check it's inside a message, not in the UI chrome
      if (!isInsideMessage(svg)) continue;

      try {
        const serializer = new XMLSerializer();
        const svgStr = serializer.serializeToString(svg);
        const dataUrl = 'data:image/svg+xml;base64,' + btoa(unescape(encodeURIComponent(svgStr)));

        const svgId = `svg_${svg.clientWidth}x${svg.clientHeight}_${svgStr.length}`;
        if (seenUrls.has(svgId)) continue;
        seenUrls.add(svgId);

        images.push({
          url: dataUrl,
          type: 'chart',
          alt: `SVG chart (${svg.clientWidth}x${svg.clientHeight})`,
          width: svg.clientWidth,
          height: svg.clientHeight,
          conversationTitle,
          conversationUrl: location.href,
          isDataUrl: true,
        });
      } catch {}
    }

    // ─── 5. Links to image files ─────────────────────────────────────
    for (const a of document.querySelectorAll('a[href]')) {
      const href = a.href;
      if (!href || seenUrls.has(href)) continue;
      if (/\.(png|jpg|jpeg|gif|webp|svg|bmp|tiff)(\?|$)/i.test(href)) {
        if (isUIUrl(href)) continue;
        seenUrls.add(href);
        images.push({
          url: href,
          type: classifyImage(href, a),
          alt: a.textContent?.trim() || '',
          width: 0,
          height: 0,
          conversationTitle,
          conversationUrl: location.href,
        });
      }
    }

    console.log(`[Collector] Found ${images.length} images, converting to data URLs...`);

    // Convert remote URLs to data URLs while we still have auth cookies.
    // ChatGPT image URLs expire — if we don't capture them now, they're gone.
    const captured = await captureImages(images);
    console.log(`[Collector] Captured ${captured.filter(i => i.dataUrl).length}/${captured.length} as data URLs`);

    return { ok: true, images: captured, title: conversationTitle };

  } catch (e) {
    console.error('[Collector] scanImages error:', e);
    return { ok: false, error: e.message };
  }
}


/**
 * Fetch remote image URLs and convert to data URLs.
 * Runs in the content script context so we have ChatGPT's session cookies.
 * Images that are already data URLs (canvas/SVG) pass through unchanged.
 */
async function captureImages(images) {
  const BATCH_SIZE = 5;  // concurrent fetches
  const results = [];

  for (let i = 0; i < images.length; i += BATCH_SIZE) {
    const batch = images.slice(i, i + BATCH_SIZE);
    const captured = await Promise.all(batch.map(async (img) => {
      // Already a data URL — keep as-is
      if (img.isDataUrl || img.url.startsWith('data:')) {
        return { ...img, dataUrl: img.url, originalUrl: img.url };
      }

      // Remote URL — fetch and convert to data URL
      try {
        const resp = await fetch(img.url, {
          credentials: 'include',
          mode: 'cors',
        });

        if (!resp.ok) {
          console.warn(`[Collector] Failed to fetch ${img.url}: ${resp.status}`);
          return { ...img, originalUrl: img.url };
        }

        const contentType = resp.headers.get('content-type') || 'image/png';

        // If the server returned JSON/HTML instead of an image, skip it
        if (contentType.includes('json') || contentType.includes('html')) {
          console.warn(`[Collector] Skipping non-image response: ${contentType} for ${img.url}`);
          return null;  // filter these out
        }

        const blob = await resp.blob();
        const dataUrl = await blobToDataUrl(blob);
        return { ...img, dataUrl, originalUrl: img.url };

      } catch (e) {
        console.warn(`[Collector] Fetch failed for ${img.url}:`, e.message);
        return { ...img, originalUrl: img.url };
      }
    }));

    results.push(...captured.filter(Boolean));
  }

  return results;
}


function blobToDataUrl(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}


// ─── Image classification ────────────────────────────────────────────

function classifyImage(url, el) {
  const u = url.toLowerCase();

  // DALL-E / OpenAI generated
  if (u.includes('oaidalleapi') || u.includes('dalle') || u.includes('openai')) {
    return 'dalle';
  }

  // User uploads via ChatGPT
  if (u.includes('oaiusercontent') || u.includes('file-') || u.includes('upload')) {
    return 'upload';
  }

  // Code interpreter / sandbox output
  if (u.includes('sandbox') || u.includes('interpreter') || u.includes('matplotlib')) {
    return 'chart';
  }

  // Data URLs from canvas/SVG extraction
  if (u.startsWith('data:')) {
    return 'chart';
  }

  // If it's a large image inside a message, likely content
  if (el) {
    const w = el.naturalWidth || el.width || el.offsetWidth || 0;
    const h = el.naturalHeight || el.height || el.offsetHeight || 0;
    if (w >= 256 && h >= 256) return 'generated';
  }

  return 'image';
}


function isUIImage(src, img) {
  // Filter out ChatGPT UI elements: avatars, icons, logos, tiny decorations
  const u = src.toLowerCase();

  // Known UI patterns
  if (u.includes('/avatars/') || u.includes('avatar')) return true;
  if (u.includes('favicon') || u.includes('logo') || u.includes('icon')) return true;
  if (u.includes('googleusercontent.com/a/')) return true; // Google profile pics
  if (u.includes('gravatar')) return true;
  if (u.includes('/assets/') && !u.includes('upload')) return true;

  // Tiny images (avatars, icons)
  const w = img.naturalWidth || img.width || 0;
  const h = img.naturalHeight || img.height || 0;
  if (w > 0 && w < 48 && h > 0 && h < 48) return true;

  // SVG data URIs that are likely icons
  if (u.startsWith('data:image/svg') && (w < 64 || h < 64)) return true;

  // Check if it's inside a user/assistant avatar container
  const parent = img.closest('[class*="avatar"], [class*="Avatar"], [data-testid*="avatar"]');
  if (parent) return true;

  return false;
}


function isUIUrl(url) {
  const u = url.toLowerCase();
  return u.includes('/avatars/') || u.includes('favicon') || u.includes('logo')
      || u.includes('icon') || u.includes('googleusercontent.com/a/')
      || u.includes('gravatar') || u.includes('emoji');
}


function isInsideMessage(el) {
  // Check if element is inside a conversation message, not UI chrome
  return !!el.closest('[data-message-id], article, [data-testid*="conversation-turn"], main, [role="main"]');
}


function getConversationTitle() {
  // Try to get the conversation title
  // 1. From the page <title>
  let title = document.title || '';
  // ChatGPT titles are like "Conversation Title — ChatGPT"
  title = title.replace(/\s*[-—|]\s*ChatGPT.*$/i, '').trim();
  if (title && title !== 'ChatGPT') return title;

  // 2. From the active sidebar item
  const activeLink = document.querySelector('nav a[class*="active"], nav a[aria-current="page"]');
  if (activeLink) return activeLink.textContent.trim();

  return 'Untitled';
}


// ═══════════════════════════════════════════════════════════════════════
//  SIDEBAR LINK EXTRACTION (shared with Anthill Spider logic)
// ═══════════════════════════════════════════════════════════════════════

/**
 * Extract conversation links via API, with optional limit.
 * Used by the Batch Scan button.
 */
async function extractSidebarLinks(limit) {
  try {
    reportProgress(0, 'Loading conversations from API...');

    const conversations = await fetchAllConversationsViaAPI();

    if (conversations && conversations.length > 0) {
      const result = limit > 0 ? conversations.slice(0, limit) : conversations;
      return { ok: true, conversations: result, total: conversations.length };
    }

    // API failed — fall back to DOM scrolling
    reportProgress(0, 'API unavailable, falling back to sidebar scan...');
    return await extractSidebarLinksViaDom(limit);
  } catch (e) {
    // Try DOM fallback on any error
    try {
      return await extractSidebarLinksViaDom(limit);
    } catch (e2) {
      return { ok: false, error: e2.message };
    }
  }
}

/** DOM-based fallback for extractSidebarLinks */
async function extractSidebarLinksViaDom(limit) {
  const sidebar = findSidebar();
  if (!sidebar) {
    return { ok: false, error: 'Could not find ChatGPT sidebar.' };
  }

  await scrollSidebarToBottom(sidebar);

  const links = extractAllLinks();
  const result = limit > 0 ? links.slice(0, limit) : links;

  return { ok: true, conversations: result, total: links.length };
}


/**
 * Find ALL conversations by querying ChatGPT's internal API directly.
 * Falls back to DOM scraping if the API is unavailable.
 */
async function findAllConversations() {
  try {
    reportProgress(0, 'Querying ChatGPT API for full conversation list...');

    const conversations = await fetchAllConversationsViaAPI();

    if (conversations && conversations.length > 0) {
      reportProgress(conversations.length, `Done! ${conversations.length} conversations found via API.`);
      return { ok: true, conversations, total: conversations.length };
    }

    // API failed — fall back to DOM scraping
    reportProgress(0, 'API unavailable, falling back to sidebar scan...');
    return await findAllConversationsViaDom();

  } catch (e) {
    console.error('[Collector] findAllConversations error:', e);
    try {
      reportProgress(0, `API error (${e.message}). Falling back to sidebar scan...`);
      return await findAllConversationsViaDom();
    } catch (e2) {
      return { ok: false, error: e2.message };
    }
  }
}


/**
 * Fetch all conversations from ChatGPT's internal API.
 * Endpoint: GET /backend-api/conversations?offset=N&limit=100&order=updated
 *
 * Since the content script runs on chatgpt.com, fetch() automatically
 * includes the session cookie — no extra auth needed.
 */
async function fetchAllConversationsViaAPI() {
  const PAGE_SIZE = 100;
  const allConversations = [];
  let offset = 0;
  let totalExpected = null;

  while (true) {
    const url = `${location.origin}/backend-api/conversations?offset=${offset}&limit=${PAGE_SIZE}&order=updated`;

    console.log(`[Collector] API fetch: offset=${offset}, have=${allConversations.length}`);

    const resp = await fetch(url, {
      method: 'GET',
      credentials: 'include',
      headers: { 'Accept': 'application/json' },
    });

    if (!resp.ok) {
      console.error(`[Collector] API returned ${resp.status}: ${resp.statusText}`);
      if (allConversations.length > 0) {
        console.log(`[Collector] API failed mid-pagination, returning ${allConversations.length} conversations`);
        return allConversations;
      }
      return null;  // Signal to use fallback
    }

    const data = await resp.json();

    if (!data || !data.items) {
      console.error('[Collector] API response has no items:', data);
      return allConversations.length > 0 ? allConversations : null;
    }

    if (totalExpected === null) {
      totalExpected = data.total || data.items.length;
      console.log(`[Collector] API reports ${totalExpected} total conversations`);
    }

    for (const item of data.items) {
      if (!item.id) continue;
      allConversations.push({
        title: item.title || 'Untitled',
        url: `${location.origin}/c/${item.id}`,
        href: `/c/${item.id}`,
      });
    }

    reportProgress(
      allConversations.length,
      `Loaded ${allConversations.length} of ${totalExpected || '?'} conversations from API...`
    );

    if (data.items.length < PAGE_SIZE) break;

    offset += data.items.length;
    if (offset > 50000) {
      console.warn('[Collector] Safety limit reached at 50000 conversations');
      break;
    }

    await sleep(200);
  }

  return allConversations;
}


/**
 * DOM-based fallback: scroll the sidebar to discover conversations.
 * Used when the API is unavailable.
 */
async function findAllConversationsViaDom() {
  const sidebar = findSidebar();
  if (!sidebar) {
    return { ok: false, error: 'Could not find ChatGPT sidebar.' };
  }

  const scrollable = findScrollableContainer(sidebar);
  if (!scrollable) {
    return { ok: false, error: 'Could not find scrollable sidebar container.' };
  }

  const maxScrollAttempts = 500;
  const scrollDelay = 600;
  let lastCount = 0;
  let stableRounds = 0;
  let lastScrollTop = -1;

  const initialCount = countSidebarLinks();
  reportProgress(initialCount, `DOM fallback: ${initialCount} visible. Scrolling...`);

  for (let i = 0; i < maxScrollAttempts; i++) {
    scrollable.scrollTop = scrollable.scrollHeight;

    const allLinks = document.querySelectorAll('nav a[href*="/c/"]');
    if (allLinks.length > 0) {
      allLinks[allLinks.length - 1].scrollIntoView({ block: 'end', behavior: 'instant' });
    }

    await sleep(scrollDelay);

    const currentCount = countSidebarLinks();
    const currentScrollTop = scrollable.scrollTop;

    if (currentCount !== lastCount) {
      stableRounds = 0;
      reportProgress(currentCount, `Found ${currentCount} conversations... (scroll ${i + 1})`);
    } else {
      stableRounds++;
      const scrollStuck = Math.abs(currentScrollTop - lastScrollTop) < 5;
      if (stableRounds >= 8 && scrollStuck) break;
      if (stableRounds >= 15) break;
    }

    lastCount = currentCount;
    lastScrollTop = currentScrollTop;
  }

  scrollable.scrollTop = 0;
  const links = extractAllLinks();
  reportProgress(links.length, `Done! ${links.length} conversations discovered (DOM).`);
  return { ok: true, conversations: links, total: links.length };
}


// ═══════════════════════════════════════════════════════════════════════
//  SIDEBAR HELPERS
// ═══════════════════════════════════════════════════════════════════════

function findSidebar() {
  return document.querySelector('nav')
      || document.querySelector('[class*="sidebar"]')
      || document.querySelector('[role="navigation"]');
}

function findScrollableContainer(sidebar) {
  // Walk up from a conversation link
  const firstLink = document.querySelector('nav a[href*="/c/"]');
  if (firstLink) {
    let el = firstLink.parentElement;
    while (el && el !== document.body) {
      const style = window.getComputedStyle(el);
      if ((style.overflowY === 'auto' || style.overflowY === 'scroll') && el.scrollHeight > el.clientHeight + 20) {
        return el;
      }
      el = el.parentElement;
    }
  }

  // Deepest scrollable div with conversation links
  const allDivs = [...sidebar.querySelectorAll('div')];
  let best = null, bestDepth = -1;
  for (const div of allDivs) {
    const style = window.getComputedStyle(div);
    if ((style.overflowY === 'auto' || style.overflowY === 'scroll') && div.scrollHeight > div.clientHeight + 20) {
      if (div.querySelector('a[href*="/c/"]')) {
        let depth = 0, p = div;
        while (p) { depth++; p = p.parentElement; }
        if (depth > bestDepth) { bestDepth = depth; best = div; }
      }
    }
  }
  if (best) return best;

  // Brute force
  for (const div of allDivs) {
    if (div.scrollHeight > div.clientHeight + 100 && div.querySelector('a[href*="/c/"]')) return div;
  }

  return sidebar;
}

async function scrollSidebarToBottom(sidebar) {
  const scrollable = findScrollableContainer(sidebar);
  const maxAttempts = 60;
  let lastCount = 0, stableCount = 0, lastScrollTop = -1;

  for (let i = 0; i < maxAttempts; i++) {
    scrollable.scrollTop = scrollable.scrollHeight;
    const allLinks = document.querySelectorAll('nav a[href*="/c/"]');
    if (allLinks.length > 0) allLinks[allLinks.length - 1].scrollIntoView({ block: 'end', behavior: 'instant' });

    await sleep(500);

    const currentCount = countSidebarLinks();
    const scrollStuck = Math.abs(scrollable.scrollTop - lastScrollTop) < 5;

    if (currentCount === lastCount) {
      stableCount++;
      if (stableCount >= 5 && scrollStuck) break;
    } else {
      stableCount = 0;
    }
    lastCount = currentCount;
    lastScrollTop = scrollable.scrollTop;
  }

  scrollable.scrollTop = 0;
}

function countSidebarLinks() {
  const seen = new Set();
  for (const sel of ['nav a[href*="/c/"]', 'nav li a[href*="/c/"]', 'nav ol a[href*="/c/"]']) {
    for (const a of document.querySelectorAll(sel)) {
      const href = a.getAttribute('href');
      if (href) seen.add(href);
    }
  }
  return seen.size;
}

function extractAllLinks() {
  const links = [], seen = new Set();
  for (const sel of ['nav a[href*="/c/"]', 'nav li a[href*="/c/"]', 'nav ol a[href*="/c/"]']) {
    for (const a of document.querySelectorAll(sel)) {
      const href = a.getAttribute('href');
      if (!href || seen.has(href)) continue;
      const title = (a.textContent || '').trim();
      if (!title || title.length > 200 || /[{}@]|keyframes|animation:|var\(/.test(title)) continue;
      seen.add(href);
      links.push({ title, url: new URL(href, location.origin).href, href });
    }
  }
  return links;
}

function reportProgress(count, message) {
  try { chrome.runtime.sendMessage({ action: 'discoveryProgress', count, message }); } catch {}
}


// ═══════════════════════════════════════════════════════════════════════
//  PAGE SCROLLING (load lazy content)
// ═══════════════════════════════════════════════════════════════════════

async function scrollToLoadAll() {
  window.scrollTo(0, 0);
  await sleep(400);

  const maxScrolls = 30;
  let lastHeight = 0, stable = 0;

  for (let i = 0; i < maxScrolls; i++) {
    window.scrollTo(0, document.body.scrollHeight);
    await sleep(500);

    const h = document.body.scrollHeight;
    if (h === lastHeight) { stable++; if (stable >= 2) break; }
    else stable = 0;
    lastHeight = h;
  }

  window.scrollTo(0, 0);
  await sleep(300);
}


function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}
