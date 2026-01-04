// Anthill Spider Content Script
(function() {
  // Prevent multiple loads
  if (window.__CHATGPT_EXPORTER_LOADED__) {
    console.log('[Spider] Already loaded, skipping');
    return;
  }
  
  window.__CHATGPT_EXPORTER_LOADED__ = true;
  console.log('[Spider] Content script loaded at', new Date().toISOString());
  
  // Message listener
  chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    console.log('[Spider] Received action:', request.action);
    
    if (request.action === 'extractLinks') {
      try {
        const conversations = extractConversationLinks();
        sendResponse({ ok: true, conversations });
      } catch (error) {
        console.error('[Spider] Error extracting links:', error);
        sendResponse({ ok: false, error: error.message });
      }
      return true;
    }
    
    if (request.action === 'scrapeThisPage') {
      console.log('[Spider] Scraping conversation:', request.conversation?.title);
      
      scrapeCurrentPage(request.conversation)
        .then(result => {
          sendResponse(result);
        })
        .catch(error => {
          console.error('[Spider] Error scraping page:', error);
          sendResponse({ ok: false, error: error.message });
        });
      
      return true; // Keep channel open for async response
    }
    
    // Handle unknown actions
    sendResponse({ ok: false, error: 'Unknown action: ' + request.action });
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
          
          let title = a.textContent?.trim() || a.innerText?.trim();
          
          // Skip CSS/JS code and very long titles
          if (!title || 
              title.length > 200 || 
              title.includes('{') || 
              title.includes('}') || 
              title.includes('px') ||
              title.includes('@keyframes') ||
              title.includes('var(') ||
              title.includes('.starburst') ||
              title.includes('animation:') ||
              title.includes('fill:') ||
              title.includes('opacity:')) {
            return;
          }
          
          title = title.replace(/\s+/g, ' ').trim();
          
          if (title.length < 2 || title.length > 150) return;
          
          conversations.push({
            id: href.replace('/c/', ''),
            title: title,
            url: 'https://chatgpt.com' + href
          });
        } catch (e) {
          console.warn('[Spider] Error processing anchor:', e);
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
      
      // Wait for page to be fully loaded
      await ensurePageFullyLoaded();
      
      // Extract messages
      const messages = extractMessages();
      
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
  
  async function ensurePageFullyLoaded() {
    console.log('[Spider] Ensuring page is fully loaded...');
    
    // First, wait for the page to be interactive
    await waitForPageInteractive();
    
    // Scroll to trigger lazy loading of all messages
    await scrollToLoadAllContent();
    
    // Wait for any animations or lazy loading to complete
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Try to find message elements
    const hasMessages = await waitForMessages();
    
    if (!hasMessages) {
      throw new Error('No messages found on page after loading');
    }
    
    console.log('[Spider] Page fully loaded and ready for scraping');
  }
  
  function waitForPageInteractive(timeout = 30000) {
    return new Promise((resolve, reject) => {
      const start = Date.now();
      
      const check = () => {
        if (document.readyState === 'complete' || document.readyState === 'interactive') {
          resolve();
        } else if (Date.now() - start > timeout) {
          reject(new Error('Timeout waiting for page to be interactive'));
        } else {
          setTimeout(check, 500);
        }
      };
      
      check();
    });
  }
  
  async function scrollToLoadAllContent() {
    console.log('[Spider] Scrolling to load all content...');
    
    // Scroll to top first
    window.scrollTo(0, 0);
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Scroll to bottom slowly to trigger lazy loading
    const scrollStep = 500;
    const scrollDelay = 300;
    const maxScrolls = 50;
    
    let lastScrollPosition = 0;
    let scrollCount = 0;
    
    return new Promise((resolve) => {
      const scrollInterval = setInterval(() => {
        // Calculate next scroll position
        const nextScroll = Math.min(
          window.scrollY + scrollStep,
          document.body.scrollHeight || document.documentElement.scrollHeight
        );
        
        // Scroll to next position
        window.scrollTo(0, nextScroll);
        
        // Check if we've reached the bottom or max scrolls
        if (nextScroll === lastScrollPosition || scrollCount >= maxScrolls) {
          clearInterval(scrollInterval);
          console.log(`[Spider] Finished scrolling after ${scrollCount} steps`);
          resolve();
        }
        
        lastScrollPosition = nextScroll;
        scrollCount++;
      }, scrollDelay);
    });
  }
  
  function waitForMessages(timeout = 15000) {
    return new Promise((resolve) => {
      const start = Date.now();
      
      const check = () => {
        // Look for message elements
        const selectors = [
          '[data-message-author-role]',
          '.markdown',
          '.whitespace-pre-wrap',
          'article',
          'div[class*="message"]',
          'div[class*="Message"]'
        ];
        
        for (const selector of selectors) {
          const elements = document.querySelectorAll(selector);
          if (elements.length > 0) {
            console.log(`[Spider] Found ${elements.length} elements with selector: ${selector}`);
            resolve(true);
            return;
          }
        }
        
        if (Date.now() - start > timeout) {
          console.warn('[Spider] Timeout waiting for messages');
          resolve(false);
          return;
        }
        
        setTimeout(check, 1000);
      };
      
      check();
    });
  }
  
  function extractMessages() {
    const messages = [];
    
    // Method 1: Look for elements with data-message-author-role (most reliable)
    const roleElements = document.querySelectorAll('[data-message-author-role]');
    
    if (roleElements.length > 0) {
      console.log(`[Spider] Found ${roleElements.length} role elements`);
      
      roleElements.forEach(element => {
        try {
          const role = element.getAttribute('data-message-author-role');
          let text = element.textContent?.trim() || element.innerText?.trim();
          
          // Clean up the text
          if (text) {
            text = text.replace(/\n+/g, ' ').replace(/\s+/g, ' ').trim();
          }
          
          if (text && text.length > 5) {
            // Skip system messages or UI elements
            if (text.includes('ChatGPT') || text.includes('Model:') || 
                text.includes('Upgrade to Plus') || text.length > 10000) {
              return;
            }
            
            messages.push({
              role: role,
              text: text,
              timestamp: new Date().toISOString(),
              source: 'role-attribute'
            });
          }
        } catch (e) {
          console.warn('[Spider] Error processing role element:', e);
        }
      });
    }
    
    // Method 2: If no messages found with role attribute, try other selectors
    if (messages.length === 0) {
      console.log('[Spider] No role elements found, trying other selectors');
      
      const fallbackSelectors = [
        '.markdown',
        '.whitespace-pre-wrap',
        'article',
        'div[class*="message"]'
      ];
      
      fallbackSelectors.forEach(selector => {
        document.querySelectorAll(selector).forEach(element => {
          try {
            let text = element.textContent?.trim() || element.innerText?.trim();
            
            if (text) {
              text = text.replace(/\n+/g, ' ').replace(/\s+/g, ' ').trim();
            }
            
            if (text && text.length > 10) {
              // Determine role based on element position or classes
              let role = 'assistant';
              if (element.closest('.items-end') || 
                  element.closest('[data-message-author-role="user"]') ||
                  element.textContent.includes('You:')) {
                role = 'user';
              }
              
              messages.push({
                role: role,
                text: text,
                source: `selector: ${selector}`
              });
            }
          } catch (e) {
            console.warn(`[Spider] Error processing ${selector} element:`, e);
          }
        });
      });
    }
    
    // Remove duplicates (same text, same role)
    const uniqueMessages = [];
    const seenTexts = new Set();
    
    messages.forEach(msg => {
      const textKey = msg.role + '|' + msg.text.substring(0, 100).replace(/\s+/g, ' ');
      if (!seenTexts.has(textKey)) {
        seenTexts.add(textKey);
        uniqueMessages.push(msg);
      }
    });
    
    console.log(`[Spider] Extracted ${uniqueMessages.length} unique messages`);
    return uniqueMessages;
  }
})();