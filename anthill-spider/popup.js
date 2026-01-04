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
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    
    const response = await chrome.tabs.sendMessage(tab.id, { action: 'extractLinks' });
    
    if (!response || !response.ok || response.conversations.length === 0) {
      alert('No conversations found. Make sure you are on ChatGPT.com with the sidebar open.');
      return;
    }
    
    // Filter out conversations with CSS code in titles (clean up)
    const cleanConversations = response.conversations.filter(conv => 
      !conv.title.includes('{') && 
      !conv.title.includes(';') && 
      !conv.title.includes('@keyframes') &&
      conv.title.length < 100
    );
    
    if (cleanConversations.length === 0) {
      alert('Found conversations but they appear to be invalid. Try refreshing the page.');
      return;
    }
    
    const proceed = confirm(`Found ${cleanConversations.length} conversations. This will open each conversation in a new tab temporarily. DO NOT CLOSE CHROME during this process. Continue?`);
    
    if (!proceed) return;
    
    // Update UI
    document.getElementById('exportBtn').style.display = 'none';
    document.getElementById('stopBtn').style.display = 'block';
    document.getElementById('progressContainer').style.display = 'block';
    
    // Start scraping with cleaned conversations
    await chrome.runtime.sendMessage({ 
      action: 'startScraping', 
      conversations: cleanConversations 
    });
    
    startStatusUpdates();
    
  } catch (error) {
    console.error('Error starting export:', error);
    alert('Error: ' + error.message);
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
    const status = await chrome.runtime.sendMessage({ action: 'getStatus' });
    
    if (status.isActive) {
      const progress = (status.completed / status.total) * 100;
      document.getElementById('progressFill').style.width = `${progress}%`;
      document.getElementById('statusText').textContent = 
        `Scraping... ${status.completed}/${status.total} conversations`;
      
      document.getElementById('exportBtn').style.display = 'none';
      document.getElementById('stopBtn').style.display = 'block';
      document.getElementById('progressContainer').style.display = 'block';
      
      if (!statusInterval) {
        startStatusUpdates();
      }
      
      // If scraping just completed, show success message
      if (status.completed >= status.total) {
        setTimeout(() => {
          alert(`Scraping complete! ${status.completed} conversations saved. File should download automatically.`);
          resetUI();
        }, 1000);
      }
    } else {
      resetUI();
    }
  } catch (error) {
    console.error('Error getting status:', error);
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
  
  if (statusInterval) {
    clearInterval(statusInterval);
    statusInterval = null;
  }
}

// Update status when popup opens
updateStatus();