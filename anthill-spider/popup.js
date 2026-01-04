// Popup script for Anthill Spider
let statusInterval = null;

document.addEventListener('DOMContentLoaded', () => {
  // Check current status
  updateStatus();
  
  // Start Export button
  document.getElementById('exportBtn').addEventListener('click', startExport);
  
  // Stop Scraping button
  document.getElementById('stopBtn').addEventListener('click', stopScraping);
  
  // Download Pipeline button
  document.getElementById('pipelineBtn').addEventListener('click', downloadPipeline);
});

async function startExport() {
  try {
    // Get current tab
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    
    // Extract conversation links from current page
    const response = await chrome.tabs.sendMessage(tab.id, { action: 'extractLinks' });
    
    if (!response || !response.ok) {
      alert('Could not extract conversations. Make sure you are on ChatGPT.com with the sidebar open.');
      return;
    }
    
    if (response.conversations.length === 0) {
      alert('No conversations found. Make sure you are viewing your ChatGPT conversation history.');
      return;
    }
    
    // Show confirmation
    const proceed = confirm(`Found ${response.conversations.length} conversations. This will open each conversation in a new tab temporarily. DO NOT CLOSE CHROME during this process. Continue?`);
    
    if (!proceed) return;
    
    // Update UI
    document.getElementById('exportBtn').style.display = 'none';
    document.getElementById('stopBtn').style.display = 'block';
    document.getElementById('progressContainer').style.display = 'block';
    
    // Start scraping
    await chrome.runtime.sendMessage({ 
      action: 'startScraping', 
      conversations: response.conversations 
    });
    
    // Start status updates
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
      // Update progress bar
      const progress = (status.completed / status.total) * 100;
      document.getElementById('progressFill').style.width = `${progress}%`;
      document.getElementById('statusText').textContent = 
        `Scraping... ${status.completed}/${status.total} conversations`;
      
      // Show stop button
      document.getElementById('exportBtn').style.display = 'none';
      document.getElementById('stopBtn').style.display = 'block';
      document.getElementById('progressContainer').style.display = 'block';
      
      // Start updates if not already
      if (!statusInterval) {
        startStatusUpdates();
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