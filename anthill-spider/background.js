// Anthill Spider Background Service Worker
console.log('[Spider] Background script loaded at', new Date().toISOString());

let scrapingSession = {
  isActive: false,
  total: 0,
  completed: 0,
  failed: 0,
  conversations: [],
  startTime: null,
  currentIndex: 0,
  processingTab: null
};

// Store active timers for cleanup
let activeTimers = new Map(); // tabId -> { tabTimeout: timeoutId, loadListener: function }

// Keep service worker alive
let keepAliveInterval = setInterval(() => {
  console.log('[Background] Keep-alive ping');
}, 20000);

// Listen for messages
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  console.log('[Background] Received message:', message.action);
  
  try {
    switch (message.action) {
      case 'startScraping':
        startScraping(message.conversations);
        sendResponse({ ok: true, message: 'Scraping started' });
        break;
        
      case 'getStatus':
        sendResponse(scrapingSession);
        break;
        
      case 'downloadPipeline':
        downloadPipeline();
        sendResponse({ ok: true, message: 'Pipeline download started' });
        break;
        
      case 'stopScraping':
        stopScraping();
        sendResponse({ ok: true, message: 'Scraping stopped' });
        break;
        
      case 'conversationScraped':
        handleScrapedConversation(message.data);
        sendResponse({ ok: true });
        break;
        
      case 'scrapingFailed':
        handleScrapingFailed();
        sendResponse({ ok: true });
        break;
        
      case 'ping':
        sendResponse({ ok: true, message: 'pong', timestamp: new Date().toISOString() });
        break;
        
      default:
        console.warn('[Background] Unknown action:', message.action);
        sendResponse({ ok: false, error: 'Unknown action: ' + message.action });
    }
  } catch (error) {
    console.error('[Background] Error handling message:', error);
    sendResponse({ ok: false, error: error.message });
  }
  
  return true;
});

function startScraping(conversations) {
  console.log(`[Background] Starting scrape of ${conversations.length} conversations`);
  
  scrapingSession = {
    isActive: true,
    total: conversations.length,
    completed: 0,
    failed: 0,
    conversations: [],
    startTime: new Date().toISOString(),
    currentIndex: 0,
    processingTab: null,
    allConversations: conversations // Store all conversations for processing
  };
  
  // Save to storage
  chrome.storage.local.set({ scrapingSession });
  
  // Start processing first conversation
  processNextConversation();
}

function processNextConversation() {
  if (!scrapingSession.isActive) {
    console.log('[Background] Scraping stopped, not processing next');
    return;
  }
  
  // Check if we're already processing a tab
  if (scrapingSession.processingTab !== null) {
    console.log('[Background] Already processing a tab, waiting...');
    // Wait 2 seconds and try again
    setTimeout(() => processNextConversation(), 2000);
    return;
  }
  
  // Check if we're done
  if (scrapingSession.completed + scrapingSession.failed >= scrapingSession.total) {
    console.log('[Background] All conversations processed');
    completeScraping();
    return;
  }
  
  // Get next conversation to process
  const conversation = scrapingSession.allConversations[scrapingSession.currentIndex];
  
  if (!conversation) {
    console.log('[Background] No more conversations to process');
    completeScraping();
    return;
  }
  
  console.log(`[Background] Processing ${scrapingSession.currentIndex + 1}/${scrapingSession.total}: ${conversation.title}`);
  
  // Create a new tab for this conversation
  chrome.tabs.create({
    url: conversation.url,
    active: false
  }, (tab) => {
    console.log(`[Background] Tab created: ${tab.id} for ${conversation.title}`);
    
    scrapingSession.processingTab = tab.id;
    chrome.storage.local.set({ scrapingSession });
    
    // Listen for tab load
    const tabLoadListener = (tabId, changeInfo) => {
      if (tabId === tab.id && changeInfo.status === 'complete') {
        // Clear the tab timeout since page loaded successfully
        clearTabTimeout(tab.id);
        
        chrome.tabs.onUpdated.removeListener(tabLoadListener);
        
        console.log(`[Background] Tab ${tab.id} loaded, waiting for content...`);
        
        // Wait for page to fully render (10 seconds)
        setTimeout(() => {
          // Scroll to ensure all content loads
          chrome.scripting.executeScript({
            target: { tabId: tab.id },
            func: () => {
              // Scroll to trigger lazy loading
              window.scrollTo(0, 0);
              setTimeout(() => {
                window.scrollTo(0, document.body.scrollHeight || document.documentElement.scrollHeight);
              }, 1000);
              return true;
            }
          }).then(() => {
            // Wait after scrolling (2 seconds)
            setTimeout(() => {
              // Now scrape the page
              console.log(`[Background] Sending scrape message to tab ${tab.id}`);
              
              chrome.tabs.sendMessage(tab.id, {
                action: 'scrapeThisPage',
                conversation: conversation
              }, (response) => {
                if (chrome.runtime.lastError) {
                  console.error(`[Background] Error sending message to tab ${tab.id}:`, chrome.runtime.lastError);
                  handleScrapingFailed();
                } else if (response && response.ok) {
                  console.log(`[Background] Successfully scraped: ${conversation.title}`);
                  handleScrapedConversation(response.data);
                } else {
                  console.error(`[Background] Failed to scrape: ${conversation.title}`, response?.error);
                  handleScrapingFailed();
                }
                
                // Clean up and close the tab
                cleanupTabAndContinue(tab.id, tabLoadListener);
              });
            }, 2000); // Wait 2 seconds after scrolling
          }).catch(error => {
            console.error(`[Background] Error executing scroll script:`, error);
            handleScrapingFailed();
            cleanupTabAndContinue(tab.id, tabLoadListener);
          });
        }, 10000); // Wait 10 seconds for page to load
      }
    };
    
    chrome.tabs.onUpdated.addListener(tabLoadListener);
    
    // Set timeout for this tab (45 seconds total)
    const tabTimeout = setTimeout(() => {
      console.error(`[Background] Timeout loading tab for: ${conversation.title}`);
      chrome.tabs.onUpdated.removeListener(tabLoadListener);
      handleScrapingFailed();
      cleanupTabAndContinue(tab.id, tabLoadListener);
    }, 45000); // 45 seconds total timeout
    
    // Store the timer and listener
    activeTimers.set(tab.id, {
      tabTimeout,
      tabLoadListener
    });
  });
}

function cleanupTabAndContinue(tabId, tabLoadListener) {
  // Clear the timeout and remove from active timers
  clearTabTimeout(tabId);
  
  // Remove the load listener if it's still attached
  if (tabLoadListener) {
    chrome.tabs.onUpdated.removeListener(tabLoadListener);
  }
  
  // Close the tab
  chrome.tabs.remove(tabId, () => {
    if (chrome.runtime.lastError) {
      console.log(`[Background] Tab ${tabId} already closed or doesn't exist`);
    }
    
    // Clear the processing tab
    scrapingSession.processingTab = null;
    
    // Move to next conversation
    scrapingSession.currentIndex++;
    chrome.storage.local.set({ scrapingSession });
    
    // Wait a moment, then process next
    setTimeout(() => {
      processNextConversation();
    }, 3000); // 3-second delay between conversations
  });
}

function clearTabTimeout(tabId) {
  if (activeTimers.has(tabId)) {
    const timerData = activeTimers.get(tabId);
    if (timerData.tabTimeout) {
      clearTimeout(timerData.tabTimeout);
    }
    activeTimers.delete(tabId);
    console.log(`[Background] Cleared timeout for tab ${tabId}`);
  }
}

function handleScrapedConversation(data) {
  scrapingSession.completed++;
  scrapingSession.conversations.push(data);
  
  // Update storage
  chrome.storage.local.set({ scrapingSession });
  
  console.log(`[Background] Progress: ${scrapingSession.completed}/${scrapingSession.total}`);
}

function handleScrapingFailed() {
  scrapingSession.failed++;
  chrome.storage.local.set({ scrapingSession });
  console.log(`[Background] Failed count: ${scrapingSession.failed}/${scrapingSession.total}`);
}

function completeScraping() {
  if (!scrapingSession.isActive) {
    console.log('[Background] Scraping already completed or stopped');
    return;
  }
  
  // Double-check we're really done
  const processed = scrapingSession.completed + scrapingSession.failed;
  if (processed < scrapingSession.total) {
    console.log(`[Background] Not ready to complete: ${processed}/${scrapingSession.total} processed`);
    setTimeout(() => completeScraping(), 1000);
    return;
  }
  
  if (scrapingSession.processingTab !== null) {
    console.log(`[Background] Still processing tab ${scrapingSession.processingTab}, waiting...`);
    setTimeout(() => completeScraping(), 1000);
    return;
  }
  
  scrapingSession.isActive = false;
  scrapingSession.endTime = new Date().toISOString();
  
  console.log('[Background] All conversations scraped, compiling dataset...');
  console.log(`[Background] Stats: ${scrapingSession.completed} completed, ${scrapingSession.failed} failed`);
  
  if (scrapingSession.conversations.length === 0) {
    console.error('[Background] No conversations were scraped successfully');
    return;
  }
  
  // Create the dataset - ALL conversations in one file
  const dataset = {
    metadata: {
      exportDate: new Date().toISOString(),
      totalConversations: scrapingSession.total,
      successfullyScraped: scrapingSession.completed,
      failed: scrapingSession.failed,
      source: 'Anthill Spider v1.0',
      startTime: scrapingSession.startTime,
      endTime: scrapingSession.endTime
    },
    conversations: scrapingSession.conversations
  };
  
  // Convert to JSON
  const jsonData = JSON.stringify(dataset, null, 2);
  
  // Create a Blob
  const blob = new Blob([jsonData], { type: 'application/json' });
  
  // Convert blob to base64 data URL (works in service workers)
  const reader = new FileReader();
  reader.onloadend = function() {
    const base64data = reader.result;
    
  // Generate consistent filename
  const filename = `chatgpt_conversations_latest.json`;
    
    // Download the file - ONE FILE with ALL conversations
    chrome.downloads.download({
      url: base64data,
      filename: filename,
      saveAs: true
    }, (downloadId) => {
      if (chrome.runtime.lastError) {
        console.error('[Background] Download failed:', chrome.runtime.lastError);
      } else {
        console.log(`[Background] Download started: ${filename} (ID: ${downloadId})`);
        console.log(`[Background] File contains ${scrapingSession.conversations.length} conversations`);
      }
      
      // Clean up storage after download
      setTimeout(() => {
        chrome.storage.local.remove(['scrapingSession']);
        console.log('[Background] Cleaned up storage');
      }, 5000);
    });
  };
  
  reader.readAsDataURL(blob);
}

function stopScraping() {
  scrapingSession.isActive = false;
  chrome.storage.local.set({ scrapingSession });
  console.log('[Background] Scraping stopped by user');
  
  // Clean up active timers
  activeTimers.forEach((timerData, tabId) => {
    if (timerData.tabTimeout) {
      clearTimeout(timerData.tabTimeout);
    }
    if (timerData.tabLoadListener) {
      chrome.tabs.onUpdated.removeListener(timerData.tabLoadListener);
    }
  });
  activeTimers.clear();
  
  // Close any open tab
  if (scrapingSession.processingTab) {
    chrome.tabs.remove(scrapingSession.processingTab, () => {
      console.log(`[Background] Closed processing tab ${scrapingSession.processingTab}`);
      scrapingSession.processingTab = null;
    });
  }
}

function downloadPipeline() {
  chrome.downloads.download({
    url: chrome.runtime.getURL('pipeline/anthill_loom_pipeline.zip'),
    filename: 'anthill_loom_pipeline.zip',
    saveAs: true
  }, (downloadId) => {
    if (chrome.runtime.lastError) {
      console.error('[Background] Pipeline download failed:', chrome.runtime.lastError);
    } else {
      console.log(`[Background] Pipeline download started with ID: ${downloadId}`);
    }
  });
}

// Load saved session on startup
chrome.storage.local.get(['scrapingSession'], (result) => {
  if (result.scrapingSession) {
    scrapingSession = result.scrapingSession;
    console.log('[Background] Loaded previous scraping session');
  }
});

// Clean up on unload
chrome.runtime.onSuspend.addListener(() => {
  console.log('[Background] Service worker is being suspended');
  clearInterval(keepAliveInterval);
  
  // Clean up any remaining timers
  activeTimers.forEach((timerData, tabId) => {
    if (timerData.tabTimeout) {
      clearTimeout(timerData.tabTimeout);
    }
  });
  activeTimers.clear();
});