// Anthill Spider Background Service Worker
console.log('[Spider] Background script loaded');

let scrapingSession = {
  isActive: false,
  total: 0,
  completed: 0,
  failed: 0,
  conversations: []
};

// Listen for messages
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  console.log('[Background] Received:', message.action);
  
  switch (message.action) {
    case 'startScraping':
      startScraping(message.conversations);
      sendResponse({ ok: true });
      break;
      
    case 'getStatus':
      sendResponse(scrapingSession);
      break;
      
    case 'downloadPipeline':
      downloadPipeline();
      sendResponse({ ok: true });
      break;
      
    case 'stopScraping':
      stopScraping();
      sendResponse({ ok: true });
      break;
      
    case 'conversationScraped':
      handleScrapedConversation(message.data);
      sendResponse({ ok: true });
      break;
      
    case 'scrapingFailed':
      scrapingSession.failed++;
      checkCompletion();
      sendResponse({ ok: true });
      break;
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
    conversations: []
  };
  
  // Process conversations one by one
  processNextConversation(conversations, 0);
}

function processNextConversation(conversations, index) {
  if (index >= conversations.length) {
    console.log('[Background] All conversations processed');
    checkCompletion();
    return;
  }
  
  if (!scrapingSession.isActive) {
    console.log('[Background] Scraping stopped by user');
    return;
  }
  
  const conversation = conversations[index];
  console.log(`[Background] Processing ${index + 1}/${conversations.length}: ${conversation.title.substring(0, 50)}...`);
  
  // Create a new tab for this conversation
  chrome.tabs.create({
    url: conversation.url,
    active: false
  }, (tab) => {
    // Wait for tab to load
    setTimeout(() => {
      chrome.tabs.sendMessage(tab.id, {
        action: 'scrapeThisPage',
        conversation: conversation
      }, (response) => {
        if (response && response.ok) {
          console.log(`[Background] Successfully scraped: ${conversation.title.substring(0, 50)}...`);
          handleScrapedConversation(response.data);
        } else {
          console.error(`[Background] Failed to scrape: ${conversation.title}`);
          scrapingSession.failed++;
        }
        
        // Close the tab
        chrome.tabs.remove(tab.id);
        
        // Process next conversation
        setTimeout(() => {
          processNextConversation(conversations, index + 1);
        }, 1000);
      });
    }, 3000); // Wait 3 seconds for page to load
  });
}

function handleScrapedConversation(data) {
  scrapingSession.completed++;
  scrapingSession.conversations.push(data);
  
  console.log(`[Background] Progress: ${scrapingSession.completed}/${scrapingSession.total}`);
  
  // Check if all done
  if (scrapingSession.completed + scrapingSession.failed >= scrapingSession.total) {
    completeScraping();
  }
}

function completeScraping() {
  console.log('[Background] All conversations scraped, compiling dataset...');
  scrapingSession.isActive = false;
  
  // Create the dataset
  const dataset = {
    metadata: {
      exportDate: new Date().toISOString(),
      totalConversations: scrapingSession.total,
      successfullyScraped: scrapingSession.completed,
      failed: scrapingSession.failed,
      source: 'Anthill Spider v1.0'
    },
    conversations: scrapingSession.conversations
  };
  
  // Convert to JSON
  const jsonData = JSON.stringify(dataset, null, 2);
  
  // Create a Blob and download it
  const blob = new Blob([jsonData], { type: 'application/json' });
  
  // Create object URL - FIXED: Use self.URL for service worker context
  const url = self.URL.createObjectURL(blob);
  
  // Download the file
  chrome.downloads.download({
    url: url,
    filename: `chatgpt_conversations_${Date.now()}.json`,
    saveAs: false // Set to true if you want "Save As" dialog
  }, (downloadId) => {
    if (chrome.runtime.lastError) {
      console.error('[Background] Download failed:', chrome.runtime.lastError);
    } else {
      console.log(`[Background] Download started with ID: ${downloadId}`);
    }
    
    // Clean up the URL object after download
    setTimeout(() => {
      self.URL.revokeObjectURL(url);
    }, 10000);
  });
}

function checkCompletion() {
  if (scrapingSession.completed + scrapingSession.failed >= scrapingSession.total) {
    completeScraping();
  }
}

function stopScraping() {
  scrapingSession.isActive = false;
  console.log('[Background] Scraping stopped by user');
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