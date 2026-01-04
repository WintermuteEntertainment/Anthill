// Anthill Spider Content Script
// Wrapped in IIFE to avoid scope issues
(function() {
  // Prevent multiple loads
  if (window.__CHATGPT_EXPORTER_LOADED__) {
    console.log('[Spider] Already loaded, skipping');
    return;
  }
  
  window.__CHATGPT_EXPORTER_LOADED__ = true;
  console.log('[Spider] Content script loaded');
  
  // Message listener
  chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    console.log('[Spider] Received action:', request.action);
    
    if (request.action === 'extractLinks') {
      const conversations = extractConversationLinks();
      sendResponse({ ok: true, conversations });
    }
    else if (request.action === 'scrapeThisPage') {
      scrapeCurrentPage(request.conversation).then(result => {
        sendResponse(result);
      }).catch(error => {
        sendResponse({ ok: false, error: error.message });
      });
      return true; // Keep channel open for async
    }
    else if (request.action === 'test') {
      sendResponse({ ok: true, message: 'Content script is working' });
    }
    
    return true;
  });
  
  function extractConversationLinks() {
    try {
      const anchors = document.querySelectorAll('a[href^="/c/"]');
      const conversations = [];
      const seen = new Set();
      
      anchors.forEach(a => {
        try {
          const href = a.getAttribute('href');
          if (!href || seen.has(href)) return;
          seen.add(href);
          
          const title = a.textContent?.trim() || a.innerText?.trim();
          if (!title) return;
          
          conversations.push({
            id: href.replace('/c/', ''),
            title: title,
            url: 'https://chatgpt.com' + href
          });
        } catch (e) {
          console.warn('[Spider] Error processing link:', e);
        }
      });
      
      console.log(`[Spider] Found ${conversations.length} conversations`);
      return conversations;
    } catch (error) {
      console.error('[Spider] Error extracting links:', error);
      return [];
    }
  }
  
  async function scrapeCurrentPage(conversation) {
    try {
      console.log(`[Spider] Scraping: ${conversation.title}`);
      
      // Wait for page to be ready
      await waitForPageReady();
      
      // Extract messages
      const messages = await extractMessages();
      
      return {
        ok: true,
        data: {
          id: conversation.id,
          title: conversation.title,
          url: conversation.url,
          messages: messages,
          scrapeDate: new Date().toISOString(),
          messageCount: messages.length
        }
      };
    } catch (error) {
      console.error(`[Spider] Error scraping ${conversation.title}:`, error);
      return {
        ok: false,
        error: error.message,
        data: {
          id: conversation.id,
          title: conversation.title,
          url: conversation.url,
          messages: [],
          error: error.message
        }
      };
    }
  }
  
  async function waitForPageReady(timeout = 10000) {
    const start = Date.now();
    
    // Check if we're on a conversation page
    if (!window.location.pathname.includes('/c/')) {
      throw new Error('Not a conversation page');
    }
    
    // Wait for message elements to appear
    while (Date.now() - start < timeout) {
      // Check for various ChatGPT message selectors
      const selectors = [
        '[data-message-author-role]',
        '.markdown',
        '.whitespace-pre-wrap',
        'article',
        'div[class*="message"]'
      ];
      
      for (const selector of selectors) {
        const elements = document.querySelectorAll(selector);
        if (elements.length > 0) {
          console.log(`[Spider] Found ${elements.length} elements with selector: ${selector}`);
          return true;
        }
      }
      
      await new Promise(resolve => setTimeout(resolve, 500));
    }
    
    throw new Error('Timeout waiting for page to load');
  }
  
  async function extractMessages() {
    const messages = [];
    
    // Method 1: Look for elements with data-message-author-role
    const roleElements = document.querySelectorAll('[data-message-author-role]');
    
    if (roleElements.length > 0) {
      console.log(`[Spider] Found ${roleElements.length} role elements`);
      
      roleElements.forEach(element => {
        try {
          const role = element.getAttribute('data-message-author-role');
          const text = element.textContent?.trim() || element.innerText?.trim();
          
          if (text && text.length > 5) {
            messages.push({
              role: role,
              text: text,
              timestamp: new Date().toISOString()
            });
          }
        } catch (e) {
          console.warn('[Spider] Error processing role element:', e);
        }
      });
    }
    
    // Method 2: If no messages found, try fallback
    if (messages.length === 0) {
      console.log('[Spider] Using fallback extraction');
      return extractMessagesFallback();
    }
    
    // Filter out duplicates and short messages
    const filteredMessages = [];
    const seenTexts = new Set();
    
    messages.forEach(msg => {
      const text = msg.text.trim();
      if (text.length > 10 && !seenTexts.has(text)) {
        seenTexts.add(text);
        filteredMessages.push(msg);
      }
    });
    
    console.log(`[Spider] Extracted ${filteredMessages.length} messages`);
    return filteredMessages;
  }
  
  function extractMessagesFallback() {
    const messages = [];
    
    // Try to find message-like divs
    const allDivs = document.querySelectorAll('div');
    
    allDivs.forEach(div => {
      try {
        const text = div.textContent?.trim() || div.innerText?.trim();
        if (!text || text.length < 20 || text.length > 5000) return;
        
        // Skip UI elements
        const skipPatterns = [
          'ChatGPT', 'Model:', 'Upgrade', 'Share', 'Export',
          'Copy', 'Regenerate', 'Continue', 'New Chat',
          'Settings', 'Dark mode', 'Log out'
        ];
        
        if (skipPatterns.some(pattern => text.includes(pattern))) return;
        
        // Determine role
        let role = 'assistant';
        if (div.closest('.items-end') || 
            div.closest('[data-message-author-role="user"]') ||
            text.includes('You:')) {
          role = 'user';
        }
        
        messages.push({
          role: role,
          text: text,
          source: 'fallback'
        });
      } catch (e) {
        // Skip errors
      }
    });
    
    // Remove duplicates
    const uniqueMessages = [];
    const seen = new Set();
    
    messages.forEach(msg => {
      const key = msg.role + '|' + msg.text.substring(0, 100);
      if (!seen.has(key)) {
        seen.add(key);
        uniqueMessages.push(msg);
      }
    });
    
    return uniqueMessages;
  }
  
  // Debug function
  window.debugSpider = async function() {
    console.log('[Spider] === DEBUG ===');
    console.log('URL:', window.location.href);
    
    const links = extractConversationLinks();
    console.log('Conversations found:', links.length);
    
    const messages = await extractMessages();
    console.log('Messages found:', messages.length);
    
    return { links, messages };
  };
})();