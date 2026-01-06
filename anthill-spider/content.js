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
      // Use async function to handle the promise
      (async () => {
        try {
          // First, scroll sidebar to load ALL conversations
          await scrollSidebarToLoadAllConversations();
          
          // Now extract all conversations
          const conversations = extractConversationLinks();
          sendResponse({ ok: true, conversations });
        } catch (error) {
          console.error('[Spider] Error in extractLinks:', error);
          // Still try to extract what we have
          const conversations = extractConversationLinks();
          sendResponse({ ok: true, conversations, warning: error.message });
        }
      })();
      
      return true; // Keep channel open for async response
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
  
  async function scrollSidebarToLoadAllConversations() {
    console.log('[Spider] Scrolling sidebar to load ALL conversations...');
    
    const sidebar = findSidebar();
    if (!sidebar) {
      console.log('[Spider] No sidebar found, skipping scrolling');
      return;
    }
    
    let lastConversationCount = 0;
    let currentConversationCount = 0;
    let scrollAttempts = 0;
    let noNewConversationsCount = 0;
    const maxNoNewConversations = 3; // Stop after 3 attempts with no new conversations
    let estimatedTotalConversations = 0;
    
    // Initial count
    lastConversationCount = countConversationsInSidebar();
    console.log(`[Spider] Initial conversation count: ${lastConversationCount}`);
    
    // Estimate total possible conversations based on scroll height
    if (sidebar.scrollHeight > 0) {
      const visibleConversations = lastConversationCount;
      const visibleHeight = sidebar.clientHeight;
      const totalHeight = sidebar.scrollHeight;
      estimatedTotalConversations = Math.round((totalHeight / visibleHeight) * visibleConversations * 0.8);
      console.log(`[Spider] Estimated total conversations: ${estimatedTotalConversations}`);
    }
    
    // Keep scrolling until no new conversations load
    while (noNewConversationsCount < maxNoNewConversations) {
      scrollAttempts++;
      
      // Scroll the sidebar
      const scrollResult = scrollSidebar(sidebar);
      if (!scrollResult) {
        console.log('[Spider] Cannot scroll sidebar further');
        break;
      }
      
      // Wait for new content to load
      // Dynamic wait time: longer if we're loading many conversations
      const baseWaitTime = 1500;
      const extraWait = scrollAttempts < 10 ? 1000 : 500;
      const waitTime = baseWaitTime + extraWait;
      
      console.log(`[Spider] Waiting ${waitTime}ms for new conversations to load (attempt ${scrollAttempts})...`);
      await new Promise(resolve => setTimeout(resolve, waitTime));
      
      // Count current conversations
      currentConversationCount = countConversationsInSidebar();
      console.log(`[Spider] After scroll ${scrollAttempts}: ${currentConversationCount} conversations`);
      
      // Check if we got new conversations
      if (currentConversationCount > lastConversationCount) {
        const newConvos = currentConversationCount - lastConversationCount;
        console.log(`[Spider] Loaded ${newConvos} new conversations!`);
        lastConversationCount = currentConversationCount;
        noNewConversationsCount = 0; // Reset counter
        
        // Update estimated time in console
        const estimatedTime = Math.round((scrollAttempts * waitTime) / 1000);
        console.log(`[Spider] Progress: ${lastConversationCount} conversations loaded in ${estimatedTime}s`);
      } else {
        noNewConversationsCount++;
        console.log(`[Spider] No new conversations (attempt ${noNewConversationsCount}/${maxNoNewConversations})`);
        
        // Try alternative scrolling strategies
        if (noNewConversationsCount === 1) {
          console.log('[Spider] Trying alternative scroll strategies...');
          
          // Strategy 1: Scroll to different positions
          sidebar.scrollTop = sidebar.scrollHeight * 0.3;
          await new Promise(resolve => setTimeout(resolve, 1000));
          
          sidebar.scrollTop = sidebar.scrollHeight * 0.6;
          await new Promise(resolve => setTimeout(resolve, 1000));
          
          // Strategy 2: Scroll by smaller increments
          const smallScrollHeight = sidebar.clientHeight * 0.3;
          for (let i = 0; i < 3; i++) {
            sidebar.scrollTop = Math.min(
              sidebar.scrollTop + smallScrollHeight,
              sidebar.scrollHeight
            );
            await new Promise(resolve => setTimeout(resolve, 800));
          }
          
          // Re-count after alternative scrolling
          currentConversationCount = countConversationsInSidebar();
          if (currentConversationCount > lastConversationCount) {
            console.log(`[Spider] Alternative strategy loaded ${currentConversationCount - lastConversationCount} new conversations!`);
            lastConversationCount = currentConversationCount;
            noNewConversationsCount = 0;
            continue;
          }
        }
      }
      
      // Check if we've reached the bottom
      const isAtBottom = Math.abs(sidebar.scrollHeight - sidebar.scrollTop - sidebar.clientHeight) < 10;
      if (isAtBottom && noNewConversationsCount >= 1) {
        console.log('[Spider] Reached bottom of sidebar');
        break;
      }
      
      // Safety: If we're stuck in an infinite loop, break after 1000 attempts (unlikely)
      if (scrollAttempts > 1000) {
        console.warn('[Spider] Safety break after 1000 scroll attempts');
        break;
      }
    }
    
    console.log(`[Spider] Finished scrolling. Total conversations loaded: ${lastConversationCount}`);
    console.log(`[Spider] Total scroll attempts: ${scrollAttempts}`);
    
    // One final wait to ensure all content is loaded
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    return lastConversationCount;
  }
  
  function findSidebar() {
    // Try multiple selectors for the sidebar
    const selectors = [
      'nav',
      'aside',
      '[role="navigation"]',
      '[data-testid="conversation-list"]',
      '[class*="sidebar"]',
      '[class*="Sidebar"]',
      'div[class*="flex-col"]',
      'div[class*="overflow-y-auto"]',
      'div[class*="scrollable"]',
      'div[class*="conversation"]',
      'div[data-radix-scroll-area-viewport]', // ChatGPT sometimes uses this
      'div[class*="react-scroll-to-bottom"]'
    ];
    
    for (const selector of selectors) {
      const elements = document.querySelectorAll(selector);
      for (const element of elements) {
        // Check if it looks like a sidebar (has scrollable content and is on the side)
        const hasScroll = element.scrollHeight > element.clientHeight;
        const isOnSide = element.getBoundingClientRect().left < 300; // Assuming sidebar is on the left
        
        if (hasScroll && isOnSide && element.clientHeight > 300) {
          console.log(`[Spider] Found sidebar with selector: ${selector}`);
          return element;
        }
      }
    }
    
    console.log('[Spider] No sidebar found with standard selectors, trying fallback...');
    
    // Fallback: Find any scrollable container with conversation links
    const allElements = document.querySelectorAll('*');
    for (const element of allElements) {
      if (element.scrollHeight > element.clientHeight + 100) {
        // Check if it contains conversation links
        const hasConversationLinks = element.querySelectorAll('a[href^="/c/"]').length > 0;
        if (hasConversationLinks) {
          console.log('[Spider] Found sidebar via fallback (contains conversation links)');
          return element;
        }
      }
    }
    
    return null;
  }
  
  function scrollSidebar(sidebar) {
    // Check if we can scroll further
    const canScroll = sidebar.scrollHeight > sidebar.clientHeight;
    const isAtBottom = Math.abs(sidebar.scrollHeight - sidebar.scrollTop - sidebar.clientHeight) < 10;
    
    if (!canScroll || isAtBottom) {
      return false;
    }
    
    // Save current position
    const previousScrollTop = sidebar.scrollTop;
    
    // Scroll by 80% of the viewport height
    const scrollAmount = sidebar.clientHeight * 0.8;
    const targetScroll = Math.min(previousScrollTop + scrollAmount, sidebar.scrollHeight);
    
    sidebar.scrollTop = targetScroll;
    
    console.log(`[Spider] Scrolled sidebar from ${previousScrollTop} to ${targetScroll}`);
    return true;
  }
  
  function countConversationsInSidebar() {
    // Count unique conversation links in the entire document
    const anchors = document.querySelectorAll('a[href^="/c/"]');
    const seen = new Set();
    
    anchors.forEach(a => {
      const href = a.getAttribute('href');
      if (href) {
        seen.add(href);
      }
    });
    
    return seen.size;
  }
  
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