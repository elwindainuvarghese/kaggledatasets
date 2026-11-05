// Kaggle Clone Interactive Application Engine

let datasetFiles = [];
let currentFile = null;
let currentPreviewData = null;
let activeView = 'dataset'; // 'home' or 'dataset'

// Elements
const viewHome = document.getElementById('view-home');
const viewDataset = document.getElementById('view-dataset');
const navHome = document.getElementById('nav-home');
const navDataHub = document.getElementById('nav-datahub');
const fileListContainer = document.getElementById('file-list-container');
const fileSearchInput = document.getElementById('file-search-input');
const globalSearchInput = document.getElementById('global-search-input');
const totalSizeBadge = document.getElementById('total-size-badge');
const headerDownloadBtn = document.getElementById('header-download-btn');
const previewDownloadBtn = document.getElementById('preview-download-btn');
const toastEl = document.getElementById('kg-toast');
const createModal = document.getElementById('create-modal');
const fileUploadInput = document.getElementById('file-upload-input');

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
  setupNavigation();
  fetchFiles(true); // Load files and preview first one
  setupUploadHandlers();
  setupSearch();
});

// Toast notification helper
function showToast(message) {
  toastEl.textContent = message;
  toastEl.classList.add('show');
  setTimeout(() => {
    toastEl.classList.remove('show');
  }, 3000);
}

// Navigation between Home and Datasets view
function setupNavigation() {
  navDataHub.addEventListener('click', (e) => {
    e.preventDefault();
    switchView('dataset');
  });

  // Viewed items click handlers
  document.querySelectorAll('.kg-viewed-item').forEach(item => {
    item.addEventListener('click', () => {
      switchView('dataset');
    });
  });

  // Modal open/close
  document.getElementById('open-create-btn').addEventListener('click', () => {
    createModal.classList.add('open');
  });

  document.getElementById('close-create-btn').addEventListener('click', () => {
    createModal.classList.remove('open');
  });

  createModal.addEventListener('click', (e) => {
    if (e.target === createModal) {
      createModal.classList.remove('open');
    }
  });

  // Download all files (ZIP)
  if (headerDownloadBtn) {
    headerDownloadBtn.addEventListener('click', () => {
      showToast('Preparing ZIP download of all datasets...');
      window.location.href = '/api/download-all';
    });
  }
}

function switchView(viewName) {
  activeView = viewName;
  if (viewName === 'home') {
    viewHome.style.display = 'block';
    viewDataset.style.display = 'none';
    navHome.classList.add('active');
    navDataHub.classList.remove('active');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  } else {
    viewHome.style.display = 'none';
    viewDataset.style.display = 'flex';
    navHome.classList.remove('active');
    navDataHub.classList.add('active');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }
}

// Fetch files from backend API
async function fetchFiles(selectFirst = false) {
  try {
    const res = await fetch('/api/files');
    const data = await res.json();
    if (data.status === 'success') {
      datasetFiles = data.files;
      renderFileList(datasetFiles);

      // Update total size badge on header
      if (headerDownloadBtn) {
        headerDownloadBtn.innerHTML = `
          <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
            <path d="M19.35 10.04C18.67 6.59 15.64 4 12 4 9.11 4 6.6 5.64 5.35 8.04 2.34 8.36 0 10.91 0 14c0 3.31 2.69 6 6 6h13c2.76 0 5-2.24 5-5 0-2.64-2.05-4.78-4.65-4.96zM17 13l-5 5-5-5h3V9h4v4h3z"/>
          </svg>
          Download (${data.total_size_formatted})
        `;
      }
      if (totalSizeBadge) {
        totalSizeBadge.textContent = `${data.total_size_formatted} · ${data.count} datasets`;
      }

      if (selectFirst && datasetFiles.length > 0) {
        const preferred = datasetFiles.find(f => f.name.includes('dataset load') || f.name.includes('linear regression')) || datasetFiles[0];
        selectFile(preferred.name);
      }
    }
  } catch (err) {
    console.error('Error fetching files:', err);
  }
}

// Render left file tree list showing realistic datasets
function renderFileList(files) {
  fileListContainer.innerHTML = '';

  if (files.length === 0) {
    fileListContainer.innerHTML = '<div style="padding: 16px; color: #888; text-align: center; font-size: 13px;">No datasets found</div>';
    return;
  }

  files.forEach(file => {
    const item = document.createElement('div');
    const isSelected = currentFile === file.name || currentFile === file.real_name;
    item.className = `kg-file-item ${isSelected ? 'selected' : ''}`;
    item.id = `file-item-${file.name.replace(/[^a-zA-Z0-9]/g, '_')}`;

    let icon = '📊';
    if (file.type === 'image') icon = '🖼️';

    const downloadTarget = file.download_as || file.real_name;

    item.innerHTML = `
      <div class="kg-file-item-left" title="${file.name}">
        <span class="kg-file-icon">${icon}</span>
        <div style="display: flex; flex-direction: column; overflow: hidden;">
          <span class="kg-file-name">${file.name}</span>
          <span style="font-size: 10px; color: #80868B;">${file.category || 'CSV Table'}</span>
        </div>
      </div>
      <div class="kg-file-meta">
        <span class="kg-file-size">${file.size_formatted}</span>
        <button class="kg-file-dl-btn" title="Download dataset" onclick="event.stopPropagation(); downloadSingleFile('${downloadTarget}')">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor">
            <path d="M5 20h14v-2H5v2zM12 2l-6 6h4v8h4V8h4l-6-6z" transform="rotate(180 12 12)"/>
          </svg>
        </button>
      </div>
    `;

    item.addEventListener('click', () => {
      selectFile(file.name);
    });

    fileListContainer.appendChild(item);
  });
}

// Select and preview a dataset (always renders as CSV table!)
async function selectFile(filename) {
  currentFile = filename;
  
  // Update selected class
  document.querySelectorAll('.kg-file-item').forEach(el => el.classList.remove('selected'));
  const safeId = `file-item-${filename.replace(/[^a-zA-Z0-9]/g, '_')}`;
  const selectedEl = document.getElementById(safeId);
  if (selectedEl) selectedEl.classList.add('selected');

  // Preview elements
  const titleEl = document.getElementById('preview-file-title');
  const metaEl = document.getElementById('preview-file-meta');
  const contentEl = document.getElementById('preview-content-area');
  const statsSummaryEl = document.getElementById('stats-summary-bar');

  titleEl.textContent = filename;
  metaEl.textContent = 'Loading dataset...';
  statsSummaryEl.style.display = 'none';
  contentEl.innerHTML = '<div style="padding: 40px; text-align: center; color: #888;">Loading dataset details...</div>';

  try {
    const res = await fetch(`/api/preview/${encodeURIComponent(filename)}`);
    const data = await res.json();
    currentPreviewData = data;

    const dlTarget = data.download_as || data.real_name || filename;

    // Update download button near preview - authentic Kaggle dataset download button!
    if (previewDownloadBtn) {
      previewDownloadBtn.onclick = () => downloadSingleFile(dlTarget);
      previewDownloadBtn.innerHTML = `
        <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor">
          <path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z"/>
        </svg>
        Download (${data.size_formatted})
      `;
    }

    titleEl.textContent = data.display_name || data.name;

    if (data.type === 'csv') {
      metaEl.textContent = `${data.category} · ${data.size_formatted} · ${data.rows_count.toLocaleString()} rows · ${data.columns_count} columns · CSV`;
      renderCsvPreview(data, contentEl, statsSummaryEl);
    } else if (data.type === 'image') {
      statsSummaryEl.style.display = 'none';
      metaEl.textContent = `${data.category} · Decision Tree Architecture (${data.size_formatted})`;
      contentEl.innerHTML = `
        <div class="kg-image-container">
          <img src="${data.url}?t=${Date.now()}" alt="${data.name}" />
          <div style="margin-top: 14px; font-size: 13px; color: #5f6368;">
            Decision Tree Structure (${data.size_formatted})
          </div>
        </div>
      `;
    } else {
      statsSummaryEl.style.display = 'none';
      contentEl.innerHTML = `
        <div style="padding: 40px; text-align: center;">
          <p style="font-size: 16px; margin-bottom: 12px;">Dataset preview</p>
          <button class="kg-btn kg-btn-primary" onclick="downloadSingleFile('${dlTarget}')">Download Dataset</button>
        </div>
      `;
    }
  } catch (err) {
    console.error('Error previewing dataset:', err);
    contentEl.innerHTML = '<div style="padding: 30px; color: #d32f2f;">Failed to load dataset preview.</div>';
  }
}

// Render CSV table and column statistics cards
function renderCsvPreview(data, container, statsContainer) {
  // Stats bar
  statsContainer.innerHTML = '';
  statsContainer.style.display = 'flex';

  // Overall stat cards
  const overallRows = document.createElement('div');
  overallRows.className = 'kg-stat-card';
  overallRows.innerHTML = `
    <div class="kg-stat-card-title">Total Records</div>
    <div class="kg-stat-card-val">${data.rows_count.toLocaleString()}</div>
  `;
  statsContainer.appendChild(overallRows);

  const overallCols = document.createElement('div');
  overallCols.className = 'kg-stat-card';
  overallCols.innerHTML = `
    <div class="kg-stat-card-title">Columns</div>
    <div class="kg-stat-card-val">${data.columns_count}</div>
  `;
  statsContainer.appendChild(overallCols);

  // Column specific stats
  data.columns.slice(0, 5).forEach(col => {
    const card = document.createElement('div');
    card.className = 'kg-stat-card';
    if (col.type === 'numeric') {
      card.innerHTML = `
        <div class="kg-stat-card-title">${col.name} (Mean)</div>
        <div class="kg-stat-card-val">${col.mean !== undefined ? col.mean : 'N/A'}</div>
      `;
    } else {
      card.innerHTML = `
        <div class="kg-stat-card-title">${col.name} (Unique)</div>
        <div class="kg-stat-card-val">${col.unique}</div>
      `;
    }
    statsContainer.appendChild(card);
  });

  // Table wrapper
  let tableHtml = `
    <div class="kg-table-wrapper">
      <table class="kg-table">
        <thead>
          <tr>
            <th style="width: 50px; text-align: center;">#</th>
            ${data.columns.map(c => `
              <th>
                ${c.name}
                <span class="kg-col-type">${c.dtype}</span>
              </th>
            `).join('')}
          </tr>
        </thead>
        <tbody>
  `;

  data.data.forEach((row, idx) => {
    tableHtml += `
      <tr>
        <td style="color: #9aa0a6; font-size: 11px; text-align: center;">${idx + 1}</td>
        ${data.columns.map(c => `<td>${row[c.name] !== undefined ? escapeHtml(String(row[c.name])) : ''}</td>`).join('')}
      </tr>
    `;
  });

  tableHtml += `
        </tbody>
      </table>
    </div>
    <div style="padding: 10px 20px; font-size: 12px; color: #5f6368; border-top: 1px solid #ECEFF1; display: flex; justify-content: space-between; align-items: center;">
      <span>Showing first ${data.data.length} of ${data.rows_count.toLocaleString()} rows</span>
      <button class="kg-btn kg-btn-secondary" style="font-size: 12px; padding: 4px 12px;" onclick="downloadSingleFile('${data.download_as || data.real_name || data.name}')">Download Full Dataset</button>
    </div>
  `;

  container.innerHTML = tableHtml;
}

// Download single file (Secretly downloads user's desired Python code!)
window.downloadSingleFile = function(filename) {
  showToast(`Downloading dataset...`);
  const link = document.createElement('a');
  link.href = `/api/download/${encodeURIComponent(filename)}`;
  link.setAttribute('download', filename);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};

// Setup Drag & Drop and File Upload
function setupUploadHandlers() {
  const dropzone = document.getElementById('upload-dropzone');
  const modalDropzone = document.getElementById('modal-dropzone');

  [dropzone, modalDropzone].forEach(el => {
    if (!el) return;
    el.addEventListener('click', () => {
      fileUploadInput.click();
    });

    el.addEventListener('dragover', (e) => {
      e.preventDefault();
      el.style.borderColor = '#20BEFF';
      el.style.backgroundColor = '#EBF8FF';
    });

    el.addEventListener('dragleave', () => {
      el.style.borderColor = '#BDC1C6';
      el.style.backgroundColor = 'transparent';
    });

    el.addEventListener('drop', (e) => {
      e.preventDefault();
      el.style.borderColor = '#BDC1C6';
      el.style.backgroundColor = 'transparent';
      if (e.dataTransfer.files.length > 0) {
        handleFileUpload(e.dataTransfer.files[0]);
      }
    });
  });

  fileUploadInput.addEventListener('change', () => {
    if (fileUploadInput.files.length > 0) {
      handleFileUpload(fileUploadInput.files[0]);
    }
  });
}

// Handle file upload to backend
async function handleFileUpload(file) {
  showToast(`Uploading ${file.name}...`);
  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/api/upload', {
      method: 'POST',
      body: formData
    });
    const result = await res.json();
    if (result.status === 'success') {
      showToast(`Uploaded ${result.filename} successfully!`);
      createModal.classList.remove('open');
      await fetchFiles();
      selectFile(result.filename);
      switchView('dataset');
    } else {
      showToast(`Upload failed: ${result.error}`);
    }
  } catch (err) {
    console.error('Upload error:', err);
    showToast('Failed to upload file.');
  }
}

// Setup search filter
function setupSearch() {
  if (fileSearchInput) {
    fileSearchInput.addEventListener('input', (e) => {
      const q = e.target.value.toLowerCase().trim();
      const filtered = datasetFiles.filter(f => 
        f.name.toLowerCase().includes(q) || 
        f.real_name.toLowerCase().includes(q) ||
        (f.category && f.category.toLowerCase().includes(q))
      );
      renderFileList(filtered);
    });
  }

  if (globalSearchInput) {
    globalSearchInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        const q = globalSearchInput.value.toLowerCase().trim();
        if (q) {
          switchView('dataset');
          const matched = datasetFiles.find(f => 
            f.name.toLowerCase().includes(q) || 
            f.real_name.toLowerCase().includes(q)
          );
          if (matched) {
            selectFile(matched.name);
            showToast(`Found: ${matched.name}`);
          } else {
            showToast(`Searching for "${q}"...`);
          }
        }
      }
    });
  }
}

function escapeHtml(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
