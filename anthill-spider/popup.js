// Popup script for Anthill Spider
let statusInterval = null;

document.addEventListener('DOMContentLoaded', () => {
  updateStatus();
  
  document.getElementById('exportBtn').addEventListener('click', startExport);
  document.getElementById('stopBtn').addEventListener('click', stopScraping);
  document.getElementById('pipelineBtn').addEventListener('click', downloadPipeline);
});

async function startExport() {
  try {
    console.log('Starting export...');
    
    // Get current tab
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    console.log('Current tab:', tab.id, tab.url);
    
    // Check if we're on ChatGPT
    if (!tab.url || (!tab.url.includes('chatgpt.com') && !tab.url.includes('chat.openai.com'))) {
      alert('Please open ChatGPT.com in this tab first.');
      return;
    }
    
    // Update UI immediately
    document.getElementById('exportBtn').style.display = 'none';
    document.getElementById('stopBtn').style.display = 'block';
    document.getElementById('progressContainer').style.display = 'block';
    document.getElementById('statusText').textContent = 'Extracting conversations...';
    
    // Try to extract links with retry logic
    let response;
    let retries = 3;
    
    for (let i = 0; i < retries; i++) {
      try {
        console.log(`Attempt ${i + 1} to extract links...`);
        response = await chrome.tabs.sendMessage(tab.id, { action: 'extractLinks' });
        
        if (response && response.ok) {
          console.log('Extract links successful:', response.conversations.length, 'conversations');
          break;
        }
      } catch (error) {
        console.log(`Attempt ${i + 1} failed:`, error.message);
        if (i < retries - 1) {
          // Wait before retrying
          await new Promise(resolve => setTimeout(resolve, 500));
        } else {
          throw new Error('Failed to extract conversations. Please refresh the ChatGPT page and try again.');
        }
      }
    }
    
    if (!response || !response.ok || !response.conversations || response.conversations.length === 0) {
      throw new Error('No conversations found. Make sure you are on ChatGPT.com with conversations in the sidebar.');
    }
    
    // Filter out bad conversations
    const cleanConversations = response.conversations.filter(conv => {
      if (!conv || !conv.title) return false;
      const title = conv.title;
      return (
        title.length > 0 &&
        title.length < 200 &&
        !title.includes('{') &&
        !title.includes('}') &&
        !title.includes('px') &&
        !title.includes('@keyframes') &&
        !title.includes('var(') &&
        !title.includes('.starburst') &&
        !title.includes('animation:')
      );
    });
    
    console.log(`Filtered: ${cleanConversations.length} valid conversations out of ${response.conversations.length}`);
    
    if (cleanConversations.length === 0) {
      throw new Error('No valid conversations found. Try scrolling the sidebar to load proper conversation titles.');
    }
    
    // REMOVED THE 5-CONVERSATION LIMIT - NOW USING ALL CONVERSATIONS
    const conversationsToScrape = cleanConversations;
    
    // Warning for large numbers of conversations
    if (conversationsToScrape.length > 50) {
      const proceed = confirm(`You are about to scrape ${conversationsToScrape.length} conversations. This will take approximately ${Math.ceil(conversationsToScrape.length * 0.5)} minutes. Continue?`);
      if (!proceed) {
        resetUI();
        return;
      }
    } else {
      const proceed = confirm(`Found ${conversationsToScrape.length} valid conversations. This will open each conversation in a new tab temporarily. DO NOT CLOSE CHROME during this process. Continue?`);
      if (!proceed) {
        resetUI();
        return;
      }
    }
    
    document.getElementById('statusText').textContent = 'Starting scraping...';
    
    // Send to background script
    console.log('Sending to background:', conversationsToScrape.length, 'conversations');
    
    // Use a promise-based approach to send message
    const sendMessageWithTimeout = (message, timeout = 5000) => {
      return new Promise((resolve, reject) => {
        const timer = setTimeout(() => {
          reject(new Error('Background script timeout'));
        }, timeout);
        
        chrome.runtime.sendMessage(message, (response) => {
          clearTimeout(timer);
          
          if (chrome.runtime.lastError) {
            reject(new Error(chrome.runtime.lastError.message));
          } else {
            resolve(response);
          }
        });
      });
    };
    
    const bgResponse = await sendMessageWithTimeout({ 
      action: 'startScraping', 
      conversations: conversationsToScrape 
    });
    
    console.log('Background response:', bgResponse);
    
    if (!bgResponse || !bgResponse.ok) {
      throw new Error('Background script failed to start scraping');
    }
    
    startStatusUpdates();
    
  } catch (error) {
    console.error('Error starting export:', error);
    alert('Error: ' + error.message);
    resetUI();
  }
}

function stopScraping() {
  if (confirm('Stop scraping? This will cancel the current operation.')) {
    chrome.runtime.sendMessage({ action: 'stopScraping' });
    resetUI();
  }
}

function downloadPipeline() {
  chrome.runtime.sendMessage({ action: 'downloadPipeline' });
}

async function updateStatus() {
  try {
    const status = await new Promise((resolve, reject) => {
      chrome.runtime.sendMessage({ action: 'getStatus' }, (response) => {
        if (chrome.runtime.lastError) {
          reject(chrome.runtime.lastError);
        } else {
          resolve(response);
        }
      });
    });
    
    if (status && status.isActive) {
      const progress = status.total > 0 ? (status.completed / status.total) * 100 : 0;
      document.getElementById('progressFill').style.width = `${progress}%`;
      
      // Show more detailed status
      let statusMessage = `Scraping... ${status.completed}/${status.total} conversations`;
      if (status.failed > 0) {
        statusMessage += ` (${status.failed} failed)`;
      }
      document.getElementById('statusText').textContent = statusMessage;
      
      document.getElementById('exportBtn').style.display = 'none';
      document.getElementById('stopBtn').style.display = 'block';
      document.getElementById('progressContainer').style.display = 'block';
      
      if (!statusInterval) {
        startStatusUpdates();
      }
      
      // If scraping just completed
      if (status.completed + status.failed >= status.total && status.total > 0) {
        setTimeout(() => {
          const successMessage = `Scraping complete! ${status.completed} conversations saved, ${status.failed} failed. File should download automatically.`;
          console.log(successMessage);
          
          // Show notification but don't block with alert during download
          document.getElementById('statusText').textContent = successMessage;
          
          setTimeout(() => {
            resetUI();
          }, 5000);
        }, 2000);
      }
    } else {
      resetUI();
    }
  } catch (error) {
    console.error('Error getting status:', error);
    resetUI();
  }
}

function startStatusUpdates() {
  if (statusInterval) clearInterval(statusInterval);
  
  statusInterval = setInterval(() => {
    updateStatus();
  }, 1000);
}

function resetUI() {
  document.getElementById('exportBtn').style.display = 'block';
  document.getElementById('stopBtn').style.display = 'none';
  document.getElementById('progressContainer').style.display = 'none';
  document.getElementById('progressFill').style.width = '0%';
  document.getElementById('statusText').textContent = '';
  
  if (statusInterval) {
    clearInterval(statusInterval);
    statusInterval = null;
  }
}

// Update status when popup opens
updateStatus();