// Popup script for Anthill Spider - Enhanced with Time Estimates
let statusInterval = null;
let sidebarScrolling = false;
let loadingMessageInterval = null;

document.addEventListener('DOMContentLoaded', () => {
  updateStatus();
  
  document.getElementById('exportBtn').addEventListener('click', startExport);
  document.getElementById('stopBtn').addEventListener('click', stopScraping);
  document.getElementById('pipelineBtn').addEventListener('click', downloadPipeline);
  
  // Add event listener for loading message updates
  document.getElementById('loadingMessage').addEventListener('click', () => {
    // Optional: Click to show more info
    console.log('Loading conversations from sidebar...');
  });
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
    
    // Show loading container
    document.getElementById('loadingContainer').style.display = 'block';
    document.getElementById('statusText').textContent = 'Starting...';
    sidebarScrolling = true;
    
    // Start animated loading message
    startLoadingMessageAnimation();
    
    // Try to extract links with retry logic
    let response;
    let retries = 3;
    
    for (let i = 0; i < retries; i++) {
      try {
        console.log(`Attempt ${i + 1} to extract links...`);
        response = await chrome.tabs.sendMessage(tab.id, { action: 'extractLinks' });
        
        if (response && response.ok) {
          console.log('Extract links successful:', response.conversations.length, 'conversations');
          if (response.warning) {
            console.warn('Sidebar scrolling warning:', response.warning);
          }
          break;
        }
      } catch (error) {
        console.log(`Attempt ${i + 1} failed:`, error.message);
        if (i < retries - 1) {
          // Update loading message
          document.getElementById('loadingMessage').textContent = `Retrying (${i + 1}/${retries})...`;
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
    
    // Use ALL conversations
    const conversationsToScrape = cleanConversations;
    
    // Hide loading container
    sidebarScrolling = false;
    stopLoadingMessageAnimation();
    document.getElementById('loadingContainer').style.display = 'none';
    
    // Calculate time estimate
    const avgTimePerConversation = 35; // seconds (conservative estimate)
    const totalSeconds = conversationsToScrape.length * avgTimePerConversation;
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    
    let timeEstimate;
    if (minutes > 0) {
      timeEstimate = `${minutes} minute${minutes > 1 ? 's' : ''}`;
      if (seconds > 30) {
        timeEstimate += ` ${seconds} seconds`;
      }
    } else {
      timeEstimate = `${seconds} seconds`;
    }
    
    const totalScrapingTime = Math.ceil(totalSeconds / 60); // In minutes for display
    
    // Different messages based on conversation count
    let confirmMessage;
    if (conversationsToScrape.length > 100) {
      confirmMessage = `Found ${conversationsToScrape.length} conversations. The scraping process will take approximately ${totalScrapingTime} minutes. This will open each conversation in a new tab temporarily. DO NOT CLOSE CHROME during this process. Continue?`;
    } else if (conversationsToScrape.length > 50) {
      confirmMessage = `Found ${conversationsToScrape.length} conversations. The scraping process will take approximately ${totalScrapingTime} minutes. This will open each conversation in a new tab temporarily. Continue?`;
    } else if (conversationsToScrape.length > 20) {
      confirmMessage = `Found ${conversationsToScrape.length} conversations. This will take approximately ${timeEstimate}. Each conversation will open in a new tab temporarily. Continue?`;
    } else {
      confirmMessage = `Found ${conversationsToScrape.length} conversations. This will take approximately ${timeEstimate}. Continue?`;
    }
    
    const proceed = confirm(confirmMessage);
    
    if (!proceed) {
      resetUI();
      return;
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

function startLoadingMessageAnimation() {
  if (loadingMessageInterval) clearInterval(loadingMessageInterval);
  
  const messages = [
    'Loading conversations from sidebar...',
    'Scrolling to load all conversations...',
    'Found more conversations...',
    'Almost done loading...'
  ];
  
  let messageIndex = 0;
  let dotCount = 0;
  
  loadingMessageInterval = setInterval(() => {
    const loadingMessage = document.getElementById('loadingMessage');
    if (loadingMessage) {
      // Rotate through messages every 5 seconds
      if (dotCount === 0) {
        messageIndex = (messageIndex + 1) % messages.length;
      }
      
      // Add animated dots
      const dots = '.'.repeat((dotCount % 3) + 1);
      loadingMessage.textContent = `${messages[messageIndex]}${dots}`;
      dotCount++;
    }
  }, 500);
}

function stopLoadingMessageAnimation() {
  if (loadingMessageInterval) {
    clearInterval(loadingMessageInterval);
    loadingMessageInterval = null;
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
      
      // Show different status if sidebar scrolling is happening
      if (sidebarScrolling) {
        // Keep loading animation running
      } else {
        let statusMessage = `Scraping... ${status.completed}/${status.total} conversations`;
        if (status.failed > 0) {
          statusMessage += ` (${status.failed} failed)`;
        }
        
        // Add estimated time remaining
        if (status.total > 0 && status.completed > 0) {
          const avgTimePer = 35; // seconds
          const remaining = status.total - status.completed;
          const estSeconds = remaining * avgTimePer;
          const estMinutes = Math.ceil(estSeconds / 60);
          
          if (estMinutes > 1) {
            statusMessage += ` • ~${estMinutes} min remaining`;
          } else if (estSeconds > 30) {
            statusMessage += ` • ~${estSeconds} sec remaining`;
          }
        }
        
        document.getElementById('statusText').textContent = statusMessage;
      }
      
      document.getElementById('exportBtn').style.display = 'none';
      document.getElementById('stopBtn').style.display = 'block';
      document.getElementById('progressContainer').style.display = 'block';
      
      if (!statusInterval) {
        startStatusUpdates();
      }
      
      // If scraping just completed
      if (status.completed + status.failed >= status.total && status.total > 0 && !sidebarScrolling) {
        setTimeout(() => {
          const successMessage = `✅ Scraped ${status.completed} conversations`;
          const failedMessage = status.failed > 0 ? ` (${status.failed} failed)` : '';
          
          document.getElementById('statusText').innerHTML = 
            `<strong>${successMessage}${failedMessage}</strong><br>File should download automatically.`;
          
          setTimeout(() => {
            resetUI();
          }, 7000);
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
  sidebarScrolling = false;
  stopLoadingMessageAnimation();
  
  document.getElementById('exportBtn').style.display = 'block';
  document.getElementById('stopBtn').style.display = 'none';
  document.getElementById('progressContainer').style.display = 'none';
  document.getElementById('loadingContainer').style.display = 'none';
  document.getElementById('progressFill').style.width = '0%';
  document.getElementById('statusText').textContent = '';
  document.getElementById('loadingMessage').textContent = 'Loading conversations from sidebar...';
  
  if (statusInterval) {
    clearInterval(statusInterval);
    statusInterval = null;
  }
}

// Update status when popup opens
updateStatus();