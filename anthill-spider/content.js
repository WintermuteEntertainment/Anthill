// content.js — Runs on ChatGPT pages
// Handles: sidebar link extraction, conversation scraping, pick-mode clicks

console.log('[Spider] Content script loaded on', location.href);

// ═══════════════════════════════════════════════════════════════════════
//  MESSAGE HANDLER
// ═══════════════════════════════════════════════════════════════════════

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  switch (msg.action) {
    case 'extractLinks':
      extractSidebarLinks(msg.limit || 0).then(sendResponse);
      return true;

    case 'findAllConversations':
      findAllConversations().then(sendResponse);
      return true;

    case 'scrapeThisPage':
      scrapeCurrentConversation(msg.conversation).then(sendResponse);
      return true;

    case 'scrapeViaAPI':
      scrapeConversationViaAPI(msg.conversationId, msg.conversation).then(sendResponse);
      return true;

    case 'enablePickMode':
      enablePickMode();
      sendResponse({ ok: true });
      return false;

    case 'disablePickMode':
      disablePickMode();
      sendResponse({ ok: true });
      return false;

    case 'ping':
      sendResponse({ ok: true });
      return false;
  }
});


// ═══════════════════════════════════════════════════════════════════════
//  SIDEBAR LINK EXTRACTION
// ═══════════════════════════════════════════════════════════════════════

/**
 * Scroll the sidebar to load all conversations, then extract links.
 * @param {number} limit - Max conversations to return (0 = all)
 */
/**
 * Extract conversation links via API, with optional limit.
 * Used by the Start Export button. Falls back to DOM if API unavailable.
 */
async function extractSidebarLinks(limit) {
  try {
    reportDiscoveryProgress(0, 'Loading conversations from API...');

    const conversations = await fetchAllConversationsViaAPI();

    if (conversations && conversations.length > 0) {
      const result = limit > 0 ? conversations.slice(0, limit) : conversations;
      console.log(`[Spider] Extracted ${result.length} conversation links (${conversations.length} total via API)`);
      return { ok: true, conversations: result, total: conversations.length };
    }

    // API failed — fall back to DOM scrolling
    reportDiscoveryProgress(0, 'API unavailable, falling back to sidebar scan...');
    return await extractSidebarLinksViaDom(limit);

  } catch (e) {
    console.error('[Spider] extractLinks error:', e);
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
    return { ok: false, error: 'Could not find ChatGPT sidebar. Make sure conversations are visible.' };
  }

  await scrollSidebarToBottom(sidebar);

  const links = extractAllLinks();
  const result = limit > 0 ? links.slice(0, limit) : links;

  console.log(`[Spider] Extracted ${result.length} conversation links (${links.length} total via DOM)`);
  return { ok: true, conversations: result, total: links.length };
}


/**
 * Find ALL conversations by querying ChatGPT's internal API directly.
 * This is far more reliable than DOM scrolling — it paginates through
 * the backend-api/conversations endpoint using the user's session cookie.
 *
 * Falls back to DOM scraping if the API is unavailable.
 */
async function findAllConversations() {
  try {
    reportDiscoveryProgress(0, 'Querying ChatGPT API for full conversation list...');

    const conversations = await fetchAllConversationsViaAPI();

    if (conversations && conversations.length > 0) {
      reportDiscoveryProgress(conversations.length, `Done! ${conversations.length} conversations found via API.`);
      return { ok: true, conversations, total: conversations.length };
    }

    // API failed — fall back to DOM scraping
    reportDiscoveryProgress(0, 'API unavailable, falling back to sidebar scan...');
    return await findAllConversationsViaDom();

  } catch (e) {
    console.error('[Spider] findAllConversations error:', e);
    // Try DOM fallback on any error
    try {
      reportDiscoveryProgress(0, `API error (${e.message}). Falling back to sidebar scan...`);
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

    console.log(`[Spider] API fetch: offset=${offset}, have=${allConversations.length}`);

    const resp = await fetch(url, {
      method: 'GET',
      credentials: 'include',
      headers: {
        'Accept': 'application/json',
      },
    });

    if (!resp.ok) {
      console.error(`[Spider] API returned ${resp.status}: ${resp.statusText}`);
      if (allConversations.length > 0) {
        // We got some — return what we have
        console.log(`[Spider] API failed mid-pagination, returning ${allConversations.length} conversations`);
        return allConversations;
      }
      return null;  // Signal to use fallback
    }

    const data = await resp.json();

    if (!data || !data.items) {
      console.error('[Spider] API response has no items:', data);
      return allConversations.length > 0 ? allConversations : null;
    }

    // First response tells us the total
    if (totalExpected === null) {
      totalExpected = data.total || data.items.length;
      console.log(`[Spider] API reports ${totalExpected} total conversations`);
    }

    // Convert API items to our link format
    for (const item of data.items) {
      if (!item.id) continue;
      const title = item.title || 'Untitled';
      const href = `/c/${item.id}`;
      const convUrl = `${location.origin}/c/${item.id}`;

      allConversations.push({ title, url: convUrl, href });
    }

    reportDiscoveryProgress(
      allConversations.length,
      `Loaded ${allConversations.length} of ${totalExpected || '?'} conversations from API...`
    );

    // Are we done?
    if (data.items.length < PAGE_SIZE) {
      // Last page — fewer items than requested means no more
      break;
    }

    offset += data.items.length;

    // Safety: don't loop forever
    if (offset > 50000) {
      console.warn('[Spider] Safety limit reached at 50000 conversations');
      break;
    }

    // Brief pause to be polite to the API
    await sleep(200);
  }

  return allConversations;
}


/**
 * DOM-based fallback: scroll the sidebar to discover conversations.
 * Used when the API is unavailable (e.g., if ChatGPT changes the endpoint).
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
  reportDiscoveryProgress(initialCount, `DOM fallback: ${initialCount} visible. Scrolling...`);

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
      reportDiscoveryProgress(currentCount, `Found ${currentCount} conversations... (scroll ${i + 1})`);
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
  reportDiscoveryProgress(links.length, `Done! ${links.length} conversations discovered (DOM).`);
  return { ok: true, conversations: links, total: links.length };
}


/**
 * Find the scrollable container that holds the conversation list.
 * Walks UP from a conversation link to find the nearest scrollable ancestor.
 */
function findScrollableContainer(sidebar) {
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

  for (const div of allDivs) {
    if (div.scrollHeight > div.clientHeight + 100 && div.querySelector('a[href*="/c/"]')) return div;
  }

  return sidebar;
}


function countSidebarLinks() {
  const seen = new Set();
  for (const sel of ['nav a[href*="/c/"]', 'nav li a[href*="/c/"]']) {
    for (const a of document.querySelectorAll(sel)) {
      const href = a.getAttribute('href');
      if (href) seen.add(href);
    }
  }
  return seen.size;
}


function extractAllLinks() {
  const links = [], seen = new Set();
  for (const sel of ['nav a[href*="/c/"]', 'nav li a[href*="/c/"]']) {
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


function reportDiscoveryProgress(count, message) {
  try { chrome.runtime.sendMessage({ action: 'discoveryProgress', count, message }); } catch {}
}


function findSidebar() {
  // Try multiple selectors for different ChatGPT UI versions
  return document.querySelector('nav')
      || document.querySelector('[class*="sidebar"]')
      || document.querySelector('[role="navigation"]');
}


async function scrollSidebarToBottom(sidebar) {
  // Use the same robust scroll container detection
  const scrollable = findScrollableContainer(sidebar);

  const maxScrollAttempts = 60;
  const scrollDelay = 500;

  let lastCount = 0;
  let stableCount = 0;
  let lastScrollTop = -1;

  for (let i = 0; i < maxScrollAttempts; i++) {
    scrollable.scrollTop = scrollable.scrollHeight;

    // Also scroll last link into view as fallback
    const allLinks = document.querySelectorAll('nav a[href*="/c/"]');
    if (allLinks.length > 0) {
      allLinks[allLinks.length - 1].scrollIntoView({ block: 'end', behavior: 'instant' });
    }

    await sleep(scrollDelay);

    const currentCount = countSidebarLinks();
    const currentScrollTop = scrollable.scrollTop;
    const scrollStuck = Math.abs(currentScrollTop - lastScrollTop) < 5;

    if (currentCount === lastCount) {
      stableCount++;
      if (stableCount >= 5 && scrollStuck) break;
    } else {
      stableCount = 0;
    }
    lastCount = currentCount;
    lastScrollTop = currentScrollTop;
  }

  // Scroll back to top so sidebar is usable
  scrollable.scrollTop = 0;
}


// ═══════════════════════════════════════════════════════════════════════
//  CONVERSATION SCRAPING
// ═══════════════════════════════════════════════════════════════════════

async function scrapeCurrentConversation(conversationMeta) {
  try {
    // Wait for the conversation to fully render
    await waitForContent();

    // Scroll to load all messages (ChatGPT lazy-loads long conversations)
    await scrollToLoadAll();

    const messages = [];

    // Find all message containers — ChatGPT uses [data-message-id] or article-like divs
    const messageEls = findMessageElements();

    for (const el of messageEls) {
      const role = detectRole(el);
      const content = extractContent(el);

      if (content.trim()) {
        messages.push({ role, content: content.trim() });
      }
    }

    if (messages.length === 0) {
      return { ok: false, error: 'No messages found on this page' };
    }

    const data = {
      title: conversationMeta?.title || document.title || 'Untitled',
      url: location.href,
      scrapedAt: new Date().toISOString(),
      messageCount: messages.length,
      messages,
    };

    console.log(`[Spider] Scraped "${data.title}": ${messages.length} messages`);
    return { ok: true, data };

  } catch (e) {
    console.error('[Spider] scrape error:', e);
    return { ok: false, error: e.message };
  }
}


function findMessageElements() {
  // Try multiple selectors for different ChatGPT versions
  let els = document.querySelectorAll('[data-message-id]');
  if (els.length > 0) return els;

  els = document.querySelectorAll('article');
  if (els.length > 0) return els;

  // Newer versions: divs with specific classes containing user/assistant messages
  els = document.querySelectorAll('[class*="message"]');
  if (els.length > 0) return els;

  // Fallback: look for the main conversation container's direct children
  const main = document.querySelector('main') || document.querySelector('[role="main"]');
  if (main) {
    // Find groups of text blocks
    els = main.querySelectorAll('[data-testid*="conversation-turn"]');
    if (els.length > 0) return els;
  }

  return [];
}


function detectRole(el) {
  // Check data attributes
  const authorAttr = el.getAttribute('data-message-author-role');
  if (authorAttr) return authorAttr;  // 'user' or 'assistant'

  // Check for avatar/icon indicators
  const text = el.textContent || '';
  const html = el.innerHTML || '';

  // ChatGPT sometimes has a heading or icon indicating role
  if (el.querySelector('[data-testid*="user"]') || html.includes('You said')) return 'user';
  if (el.querySelector('[data-testid*="assistant"]') || html.includes('ChatGPT said')) return 'assistant';

  // Check for user icon (usually a circle with initials)
  const imgs = el.querySelectorAll('img');
  for (const img of imgs) {
    const alt = (img.alt || '').toLowerCase();
    if (alt.includes('user') || alt.includes('you')) return 'user';
    if (alt.includes('gpt') || alt.includes('chatgpt') || alt.includes('assistant')) return 'assistant';
  }

  // CSS class hints
  const classes = el.className || '';
  if (/\buser\b/i.test(classes)) return 'user';
  if (/\bassistant\b/i.test(classes) || /\bbot\b/i.test(classes)) return 'assistant';

  return 'unknown';
}


function extractContent(el) {
  // Try to get the actual message content, excluding UI chrome
  // Look for markdown-rendered content container
  const contentEl = el.querySelector('[class*="markdown"]')
                 || el.querySelector('[class*="prose"]')
                 || el.querySelector('[class*="message-content"]')
                 || el.querySelector('.text-base');

  const target = contentEl || el;

  // Get text content but try to preserve code blocks
  let content = '';
  const walker = document.createTreeWalker(target, NodeFilter.SHOW_ELEMENT | NodeFilter.SHOW_TEXT);

  let node;
  while (node = walker.nextNode()) {
    if (node.nodeType === Node.TEXT_NODE) {
      content += node.textContent;
    } else if (node.nodeName === 'BR' || node.nodeName === 'P' || node.nodeName === 'DIV') {
      if (content && !content.endsWith('\n')) content += '\n';
    } else if (node.nodeName === 'PRE' || node.nodeName === 'CODE') {
      if (!content.endsWith('\n')) content += '\n';
      content += '```\n';
    }
  }

  return content;
}


async function waitForContent(timeoutMs = 8000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    if (findMessageElements().length > 0) return;
    await sleep(500);
  }
}


async function scrollToLoadAll() {
  // Scroll to top first, then bottom, to trigger lazy-loaded content
  window.scrollTo(0, 0);
  await sleep(500);

  const maxScrolls = 20;
  let lastHeight = 0;
  let stable = 0;

  for (let i = 0; i < maxScrolls; i++) {
    window.scrollTo(0, document.body.scrollHeight);
    await sleep(600);

    const h = document.body.scrollHeight;
    if (h === lastHeight) {
      stable++;
      if (stable >= 2) break;
    } else {
      stable = 0;
    }
    lastHeight = h;
  }

  // Scroll back to top
  window.scrollTo(0, 0);
  await sleep(300);
}


// ═══════════════════════════════════════════════════════════════════════
//  PICK MODE — Click a sidebar item to export just that conversation
// ═══════════════════════════════════════════════════════════════════════

let pickModeActive = false;
let pickOverlay = null;
let pickHighlight = null;

function enablePickMode() {
  if (pickModeActive) return;
  pickModeActive = true;

  // Create overlay to intercept clicks
  pickOverlay = document.createElement('div');
  pickOverlay.id = 'spider-pick-overlay';
  pickOverlay.style.cssText = `
    position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    z-index: 99998; cursor: crosshair;
    background: rgba(102, 126, 234, 0.08);
  `;

  // Highlight element
  pickHighlight = document.createElement('div');
  pickHighlight.id = 'spider-pick-highlight';
  pickHighlight.style.cssText = `
    position: fixed; z-index: 99999;
    border: 2px solid #667eea; border-radius: 6px;
    background: rgba(102, 126, 234, 0.15);
    pointer-events: none;
    transition: all 0.1s ease;
    display: none;
  `;

  document.body.appendChild(pickOverlay);
  document.body.appendChild(pickHighlight);

  pickOverlay.addEventListener('mousemove', pickMouseMove);
  pickOverlay.addEventListener('click', pickClick);
  document.addEventListener('keydown', pickKeyDown);
}

function disablePickMode() {
  pickModeActive = false;
  if (pickOverlay) { pickOverlay.remove(); pickOverlay = null; }
  if (pickHighlight) { pickHighlight.remove(); pickHighlight = null; }
  document.removeEventListener('keydown', pickKeyDown);
}

function pickMouseMove(e) {
  // Temporarily hide overlay so elementFromPoint finds the real element underneath
  pickOverlay.style.pointerEvents = 'none';
  const elUnder = document.elementFromPoint(e.clientX, e.clientY);
  pickOverlay.style.pointerEvents = 'auto';
  if (!elUnder) return;

  // Walk up to find the <a> with /c/ href
  const link = findConversationLink(elUnder);
  if (link) {
    const rect = link.getBoundingClientRect();
    pickHighlight.style.left = rect.left + 'px';
    pickHighlight.style.top = rect.top + 'px';
    pickHighlight.style.width = rect.width + 'px';
    pickHighlight.style.height = rect.height + 'px';
    pickHighlight.style.display = 'block';
  } else {
    pickHighlight.style.display = 'none';
  }
}

function pickClick(e) {
  e.preventDefault();
  e.stopPropagation();

  // Temporarily hide overlay to find element underneath
  pickOverlay.style.pointerEvents = 'none';
  const elUnder = document.elementFromPoint(e.clientX, e.clientY);
  pickOverlay.style.pointerEvents = 'auto';

  const link = findConversationLink(elUnder);
  if (link) {
    const href = link.getAttribute('href');
    const title = (link.textContent || '').trim();
    const url = new URL(href, location.origin).href;

    console.log(`[Spider] Pick mode selected: "${title}" → ${url}`);

    // Send to background for single-conversation scrape
    chrome.runtime.sendMessage({
      action: 'pickConversation',
      conversation: { title, url, href }
    });

    disablePickMode();
  }
}

function pickKeyDown(e) {
  if (e.key === 'Escape') {
    disablePickMode();
    chrome.runtime.sendMessage({ action: 'pickCancelled' });
  }
}

function findConversationLink(el) {
  let current = el;
  for (let i = 0; i < 10 && current; i++) {
    if (current.tagName === 'A') {
      const href = current.getAttribute('href') || '';
      if (href.includes('/c/')) return current;
    }
    current = current.parentElement;
  }
  return null;
}


// ─── Utility ─────────────────────────────────────────────────────────
function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}
