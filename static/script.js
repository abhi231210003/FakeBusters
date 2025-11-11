function updatePlaceholder() {
    const task = document.getElementById('task').value;
    const textarea = document.getElementById('message');
    
    // Clear textarea when switching
    textarea.value = '';
    
    // Hide previous prediction result
    const resultDiv = document.querySelector('.result');
    if (resultDiv) {
        resultDiv.style.display = 'none';
    }
    
    if (task === 'fake_news') {
        textarea.placeholder = 'Enter news headline or article here...\nExample: "Scientists discover new planet in solar system"';
    } else {
        textarea.placeholder = 'Enter email message here...\nExample: "Congratulations! You have won $1,000,000. Click here to claim your prize!"';
    }
}

function setPlaceholder() {
    const task = document.getElementById('task').value;
    const textarea = document.getElementById('message');
    
    // Only update placeholder, don't clear or hide result
    if (task === 'fake_news') {
        textarea.placeholder = 'Enter news headline or article here...\nExample: "Scientists discover new planet in solar system"';
    } else {
        textarea.placeholder = 'Enter email message here...\nExample: "Congratulations! You have won $1,000,000. Click here to claim your prize!"';
    }
}

function clearForm() {
    document.getElementById('message').value = '';
    const resultDiv = document.querySelector('.result');
    if (resultDiv) {
        resultDiv.style.display = 'none';
    }
}

// Theme Toggle Functionality
function toggleTheme() {
    const body = document.body;
    const themeLabel = document.getElementById('themeLabel');
    const icon = document.querySelector('.toggle-thumb .icon');
    
    body.classList.toggle('dark-theme');
    
    // Update label and icon
    if (body.classList.contains('dark-theme')) {
        themeLabel.textContent = 'Dark Mode';
        icon.textContent = '☀️';
        localStorage.setItem('theme', 'dark');
    } else {
        themeLabel.textContent = 'Light Mode';
        icon.textContent = '🌙';
        localStorage.setItem('theme', 'light');
    }
}

// Load saved theme or detect system preference
function loadTheme() {
    const savedTheme = localStorage.getItem('theme');
    const themeLabel = document.getElementById('themeLabel');
    const icon = document.querySelector('.toggle-thumb .icon');
    
    if (savedTheme) {
        // Use saved preference
        if (savedTheme === 'dark') {
            document.body.classList.add('dark-theme');
            themeLabel.textContent = 'Dark Mode';
            icon.textContent = '☀️';
        } else {
            // Light mode - ensure dark-theme is NOT applied
            document.body.classList.remove('dark-theme');
            themeLabel.textContent = 'Light Mode';
            icon.textContent = '🌙';
        }
    } else {
        // No saved preference - default to dark mode and save it
        document.body.classList.add('dark-theme');
        themeLabel.textContent = 'Dark Mode';
        icon.textContent = '☀️';
        localStorage.setItem('theme', 'dark');
    }
    
    // Clean up any dark-theme class from html element
    document.documentElement.classList.remove('dark-theme');
}

// Listen for system theme changes
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
    // Only auto-switch if user hasn't manually set a preference
    const themeLabel = document.getElementById('themeLabel');
    const icon = document.querySelector('.toggle-thumb .icon');
    
    if (!localStorage.getItem('theme')) {
        if (e.matches) {
            document.body.classList.add('dark-theme');
            themeLabel.textContent = 'Dark Mode';
            icon.textContent = '☀️';
        } else {
            document.body.classList.remove('dark-theme');
            themeLabel.textContent = 'Light Mode';
            icon.textContent = '🌙';
        }
    }
});

// Show loading spinner on form submit
document.getElementById('predictionForm').addEventListener('submit', function() {
    document.getElementById('loadingSpinner').style.display = 'block';
    document.getElementById('predictBtn').disabled = true;
});

// CSV form - show simple uploading indicator by disabling submit
const csvForm = document.getElementById('csvForm');
if (csvForm) {
    csvForm.addEventListener('submit', function() {
        // disable submit to avoid double uploads
        const btn = csvForm.querySelector('button[type=submit]');
        if (btn) btn.disabled = true;
    });
}

// show selected CSV filename
const csvFileInput = document.getElementById('csvFile');
const csvFilenameSpan = document.getElementById('csvFilename');
if (csvFileInput && csvFilenameSpan) {
    csvFileInput.addEventListener('change', function(e) {
        const f = e.target.files && e.target.files[0];
        if (f) {
            csvFilenameSpan.textContent = `${f.name} (${Math.round(f.size/1024)} KB)`;
        } else {
            csvFilenameSpan.textContent = 'No file chosen';
        }
    });
}

// Theme toggle button event listener
document.getElementById('themeToggle').addEventListener('click', toggleTheme);

// Tab switching logic
function switchTab(targetId) {
    const tabs = document.querySelectorAll('.tab');
    tabs.forEach(t => t.classList.remove('active'));
    const btn = Array.from(tabs).find(b => b.dataset && b.dataset.target === targetId);
    if (btn) btn.classList.add('active');

    const contents = document.querySelectorAll('.tab-content');
    contents.forEach(c => {
        if (c.id === targetId) {
            c.style.display = '';
        } else {
            c.style.display = 'none';
        }
    });

    // if switching to CSV tab, hide single-result area to avoid confusion
    if (targetId === 'csvTab') {
        const resultDiv = document.querySelector('.result');
        if (resultDiv) resultDiv.style.display = 'none';
    }
}

// Attach tab buttons
document.querySelectorAll('.tab').forEach(btn => {
    btn.addEventListener('click', (e) => {
        const target = e.currentTarget.dataset.target;
        if (target) switchTab(target);
    });
});

// Initialize placeholder on page load (without clearing)
window.onload = function() {
    // Load theme FIRST before anything else
    loadTheme();
    setPlaceholder();
    
    // Only clear form automatically when there is no server-rendered result.
    // If the server provided a result (after prediction via PRG), keep it visible.
    const resultDiv = document.querySelector('.result');
    if (!resultDiv) {
        clearForm();
    }
    
    // Tab initialization: respect server-set visibility, don't force switch
    // The server already set the correct tab active and visible via inline styles
    // We just ensure the tab buttons and content are in sync
    const activeBtn = document.querySelector('.tab.active');
    if (activeBtn && activeBtn.dataset && activeBtn.dataset.target) {
        const targetId = activeBtn.dataset.target;
        // Only adjust content visibility if it's not already correct
        const targetContent = document.getElementById(targetId);
        if (targetContent && targetContent.style.display === 'none') {
            // Server said this tab is active but client hid it - fix it
            switchTab(targetId);
        }
        // Otherwise trust the server's initial state
    }
};
