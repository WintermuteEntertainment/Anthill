// Anthill Spider Background Service Worker
console.log('[Spider] Background script loaded');

let scrapingSession = {
  isActive: false,
  total: 0,
  completed: 0,
  failed: 0,
  conversations: [],
  startTime: null,
  activeTabs: new Set()
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
    conversations: [],
    startTime: new Date().toISOString(),
    activeTabs: new Set()
  };
  
  // Save to storage for persistence
  chrome.storage.local.set({ scrapingSession });
  
  // Process each conversation with a delay
  processConversationsSequentially(conversations, 0);
}

async function processConversationsSequentially(conversations, index) {
  if (index >= conversations.length || !scrapingSession.isActive) {
    checkCompletion();
    return;
  }
  
  const conversation = conversations[index];
  console.log(`[Background] Processing ${index + 1}/${conversations.length}: ${conversation.title}`);
  
  try {
    // Create a new tab
    const tab = await chrome.tabs.create({
      url: conversation.url,
      active: false
    });
    
    scrapingSession.activeTabs.add(tab.id);
    
    // Wait for tab to load completely
    await waitForTabLoad(tab.id);
    
    // Wait a bit for content to render
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Send message to scrape this tab
    chrome.tabs.sendMessage(tab.id, {
      action: 'scrapeThisPage',
      conversation: conversation
    }, (response) => {
      if (chrome.runtime.lastError) {
        console.error(`[Background] Error sending message to tab ${tab.id}:`, chrome.runtime.lastError);
        scrapingSession.failed++;
      } else if (response && response.ok) {
        handleScrapedConversation(response.data);
      } else {
        scrapingSession.failed++;
      }
      
      // Close the tab
      chrome.tabs.remove(tab.id);
      scrapingSession.activeTabs.delete(tab.id);
      
      // Process next conversation
      setTimeout(() => {
        processConversationsSequentially(conversations, index + 1);
      }, 1000);
    });
    
  } catch (error) {
    console.error(`[Background] Error processing ${conversation.title}:`, error);
    scrapingSession.failed++;
    
    // Continue with next
    setTimeout(() => {
      processConversationsSequentially(conversations, index + 1);
    }, 1000);
  }
}

function waitForTabLoad(tabId) {
  return new Promise((resolve) => {
    const listener = (updatedTabId, changeInfo) => {
      if (updatedTabId === tabId && changeInfo.status === 'complete') {
        chrome.tabs.onUpdated.removeListener(listener);
        resolve();
      }
    };
    
    chrome.tabs.onUpdated.addListener(listener);
    
    // Timeout after 10 seconds
    setTimeout(() => {
      chrome.tabs.onUpdated.removeListener(listener);
      resolve();
    }, 10000);
  });
}

function handleScrapedConversation(data) {
  scrapingSession.completed++;
  scrapingSession.conversations.push(data);
  
  // Update storage
  chrome.storage.local.set({ scrapingSession });
  
  console.log(`[Background] Scraped ${scrapingSession.completed}/${scrapingSession.total}: ${data.title}`);
}

function checkCompletion() {
  if (scrapingSession.completed + scrapingSession.failed >= scrapingSession.total) {
    scrapingSession.isActive = false;
    scrapingSession.endTime = new Date().toISOString();
    
    // Close any remaining tabs
    scrapingSession.activeTabs.forEach(tabId => {
      chrome.tabs.remove(tabId).catch(() => {});
    });
    
    // Compile and download
    compileAndDownload();
  }
}

function stopScraping() {
  scrapingSession.isActive = false;
  
  // Close all active tabs
  scrapingSession.activeTabs.forEach(tabId => {
    chrome.tabs.remove(tabId).catch(() => {});
  });
  
  console.log('[Background] Scraping stopped by user');
}

function compileAndDownload() {
  console.log('[Background] Compiling dataset...');
  
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
  
  const jsonData = JSON.stringify(dataset, null, 2);
  const blob = new Blob([jsonData], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const filename = `chatgpt_conversations_${timestamp}.json`;
  
  chrome.downloads.download({
    url: url,
    filename: filename,
    saveAs: true
  }, (downloadId) => {
    if (chrome.runtime.lastError) {
      console.error('[Background] Download failed:', chrome.runtime.lastError);
    } else {
      console.log(`[Background] Download started: ${filename}`);
    }
    
    // Clean up
    setTimeout(() => {
      URL.revokeObjectURL(url);
      chrome.storage.local.remove(['scrapingSession']);
    }, 10000);
  });
}

function downloadPipeline() {
  chrome.downloads.download({
    url: chrome.runtime.getURL('pipeline/anthill_loom_pipeline.zip'),
    filename: 'anthill_loom_pipeline.zip',
    saveAs: true
  });
}

// Load saved session on startup
chrome.storage.local.get(['scrapingSession'], (result) => {
  if (result.scrapingSession) {
    scrapingSession = result.scrapingSession;
    scrapingSession.activeTabs = new Set(scrapingSession.activeTabs || []);
    
    // Check if we need to resume
    if (scrapingSession.isActive && 
        scrapingSession.completed + scrapingSession.failed < scrapingSession.total) {
      console.log('[Background] Resuming previous scraping session');
    }
  }
});