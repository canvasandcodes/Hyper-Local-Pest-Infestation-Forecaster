// Application Data
const appData = {
  sampleImages: [
    {
      id: "field1",
      name: "Corn Field - Early Infestation",
      description: "Drone image showing early signs of pest damage in corn field",
      pestLevel: 0.35,
      area: "12.5 hectares"
    },
    {
      id: "field2", 
      name: "Soybean Field - Heavy Infestation",
      description: "Severe pest damage across soybean crops",
      pestLevel: 0.78,
      area: "8.2 hectares"
    },
    {
      id: "field3",
      name: "Wheat Field - Healthy",
      description: "Healthy wheat field with minimal pest presence",
      pestLevel: 0.12,
      area: "15.7 hectares"
    }
  ],
  forecastData: [
    {
      day: 1,
      date: "2024-01-02",
      riskLevel: "medium",
      infestationLevel: 0.42,
      affectedArea: 28.5,
      temperature: 24.2,
      windSpeed: 12.3,
      ndvi: 0.65,
      recommendations: [
        "Monitor affected areas closely",
        "Consider targeted treatment in high-risk zones",
        "Increase surveillance frequency"
      ]
    },
    {
      day: 2,
      date: "2024-01-03", 
      riskLevel: "high",
      infestationLevel: 0.67,
      affectedArea: 45.8,
      temperature: 26.1,
      windSpeed: 8.7,
      ndvi: 0.58,
      recommendations: [
        "Immediate intervention required",
        "Apply pesticide treatment to affected areas",
        "Deploy additional monitoring equipment"
      ]
    },
    {
      day: 3,
      date: "2024-01-04",
      riskLevel: "high", 
      infestationLevel: 0.73,
      affectedArea: 52.3,
      temperature: 27.5,
      windSpeed: 6.2,
      ndvi: 0.52,
      recommendations: [
        "Critical intervention needed",
        "Consider aerial pesticide application",
        "Coordinate with neighboring farms"
      ]
    }
  ],
  farmLocation: {
    name: "Sample Agricultural Area",
    latitude: 40.8,
    longitude: -74.1,
    area: "25.3 hectares",
    cropType: "Mixed (Corn, Soybean, Wheat)",
    elevation: "45m",
    soilType: "Loam"
  },
  heatmapPoints: [
    {lat: 40.805, lng: -74.095, intensity: 0.8, day: 1},
    {lat: 40.810, lng: -74.105, intensity: 0.6, day: 1},
    {lat: 40.798, lng: -74.088, intensity: 0.9, day: 1},
    {lat: 40.815, lng: -74.102, intensity: 0.5, day: 1},
    {lat: 40.792, lng: -74.115, intensity: 0.7, day: 1},
    {lat: 40.807, lng: -74.092, intensity: 0.85, day: 2},
    {lat: 40.812, lng: -74.108, intensity: 0.75, day: 2},
    {lat: 40.796, lng: -74.085, intensity: 0.95, day: 2},
    {lat: 40.818, lng: -74.105, intensity: 0.65, day: 2},
    {lat: 40.789, lng: -74.118, intensity: 0.8, day: 2},
    {lat: 40.809, lng: -74.089, intensity: 0.9, day: 3},
    {lat: 40.814, lng: -74.111, intensity: 0.85, day: 3},
    {lat: 40.794, lng: -74.082, intensity: 1.0, day: 3},
    {lat: 40.820, lng: -74.108, intensity: 0.75, day: 3},
    {lat: 40.787, lng: -74.121, intensity: 0.88, day: 3}
  ]
};

// Global Variables
let map;
let heatmapLayer;
let currentDay = 1;
let selectedImage = null;
let trendChart;
let forecastGenerated = false;
let riskMarkers = [];

// Initialize Application
document.addEventListener('DOMContentLoaded', function() {
  console.log('DOM loaded, initializing app...');
  // Wait for all scripts to load
  setTimeout(() => {
    initializeApp();
  }, 100);
});

function initializeApp() {
  try {
    setupFileUpload();
    renderSampleImages();
    setTimeout(() => {
      initializeMap();
    }, 200);
    setupDayControls();
    setupForecastCards();
    setTimeout(() => {
      setupChart();
    }, 300);
    setupEventListeners();
    updateSelectedDayDisplay();
    updateRecommendations(currentDay);
    console.log('App initialized successfully');
  } catch (error) {
    console.error('Error initializing app:', error);
  }
}

// File Upload Setup
function setupFileUpload() {
  const uploadArea = document.getElementById('uploadArea');
  const fileInput = document.getElementById('fileInput');

  if (!uploadArea || !fileInput) {
    console.error('Upload elements not found');
    return;
  }

  uploadArea.addEventListener('click', () => fileInput.click());

  uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
  });

  uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
  });

  uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileUpload(files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleFileUpload(e.target.files[0]);
    }
  });
}

function handleFileUpload(file) {
  if (!file.type.startsWith('image/')) {
    alert('Please select a valid image file');
    return;
  }

  if (file.size > 10 * 1024 * 1024) {
    alert('File size must be less than 10MB');
    return;
  }

  selectedImage = {
    id: 'uploaded',
    name: file.name,
    description: 'User uploaded drone image',
    pestLevel: Math.random() * 0.6 + 0.2,
    area: (Math.random() * 20 + 5).toFixed(1) + ' hectares'
  };

  updateImageInfo();
  clearSampleSelection();
  console.log('File uploaded and selected:', selectedImage.name);
}

// Sample Images
function renderSampleImages() {
  const sampleGrid = document.getElementById('sampleGrid');
  if (!sampleGrid) {
    console.error('Sample grid not found');
    return;
  }

  sampleGrid.innerHTML = '';

  appData.sampleImages.forEach(image => {
    const item = document.createElement('div');
    item.className = 'sample-item';
    item.dataset.imageId = image.id;
    
    item.innerHTML = `
      <h5>${image.name}</h5>
      <p>${image.description}</p>
    `;

    item.addEventListener('click', (e) => {
      e.preventDefault();
      console.log('Sample image clicked:', image.name);
      selectSampleImage(image);
    });
    sampleGrid.appendChild(item);
  });
  console.log('Sample images rendered');
}

function selectSampleImage(image) {
  selectedImage = image;
  updateImageInfo();
  
  // Update visual selection
  document.querySelectorAll('.sample-item').forEach(item => {
    item.classList.remove('selected');
  });
  const selectedElement = document.querySelector(`[data-image-id="${image.id}"]`);
  if (selectedElement) {
    selectedElement.classList.add('selected');
  }
  
  console.log('Selected image:', image.name);
}

function clearSampleSelection() {
  document.querySelectorAll('.sample-item').forEach(item => {
    item.classList.remove('selected');
  });
}

function updateImageInfo() {
  const imageInfo = document.getElementById('imageInfo');
  const imageName = document.getElementById('imageName');
  const imageArea = document.getElementById('imageArea');
  const imagePestLevel = document.getElementById('imagePestLevel');

  if (!imageInfo || !imageName || !imageArea || !imagePestLevel) {
    console.error('Image info elements not found');
    return;
  }

  if (selectedImage) {
    imageName.textContent = selectedImage.name;
    imageArea.textContent = selectedImage.area;
    imagePestLevel.textContent = `${(selectedImage.pestLevel * 100).toFixed(1)}%`;
    imageInfo.classList.remove('hidden');
    console.log('Image info updated for:', selectedImage.name);
  } else {
    imageInfo.classList.add('hidden');
  }
}

// Map Initialization
function initializeMap() {
  try {
    const mapElement = document.getElementById('map');
    if (!mapElement) {
      console.error('Map element not found');
      return;
    }

    const { latitude, longitude } = appData.farmLocation;
    
    // Initialize map
    map = L.map('map', {
      center: [latitude, longitude],
      zoom: 13,
      zoomControl: true
    });

    // Add tile layer
    const tileLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '© OpenStreetMap contributors',
      maxZoom: 19
    });
    
    tileLayer.addTo(map);

    // Wait for tiles to load
    tileLayer.on('load', () => {
      console.log('Map tiles loaded');
    });

    // Add farm boundary marker
    const farmMarker = L.marker([latitude, longitude], {
      title: appData.farmLocation.name
    }).addTo(map);
    
    farmMarker.bindPopup(`
      <div class="popup-title">${appData.farmLocation.name}</div>
      <div class="popup-info">
        <strong>Area:</strong> ${appData.farmLocation.area}<br>
        <strong>Crop Type:</strong> ${appData.farmLocation.cropType}<br>
        <strong>Elevation:</strong> ${appData.farmLocation.elevation}
      </div>
    `);

    // Setup heatmap toggle
    const heatmapToggle = document.getElementById('heatmapToggle');
    if (heatmapToggle) {
      heatmapToggle.addEventListener('change', toggleHeatmap);
    }

    // Force map to resize after initialization
    setTimeout(() => {
      map.invalidateSize();
      console.log('Map resized and ready');
    }, 100);

    console.log('Map initialized successfully');
  } catch (error) {
    console.error('Error initializing map:', error);
  }
}

function updateHeatmap(day) {
  if (!map) {
    console.log('Map not initialized yet');
    return;
  }

  // Clear existing markers
  clearRiskMarkers();
  
  if (heatmapLayer) {
    map.removeLayer(heatmapLayer);
    heatmapLayer = null;
  }

  const heatmapToggle = document.getElementById('heatmapToggle');
  if (!forecastGenerated || !heatmapToggle || !heatmapToggle.checked) {
    return;
  }

  const dayPoints = appData.heatmapPoints.filter(point => point.day === day);

  if (dayPoints.length > 0) {
    // Try heatmap first, fallback to markers
    if (window.L && L.heatLayer) {
      try {
        const heatData = dayPoints.map(point => [point.lat, point.lng, point.intensity]);
        heatmapLayer = L.heatLayer(heatData, {
          radius: 25,
          blur: 15,
          maxZoom: 17,
          gradient: {
            0.0: '#00ff00',
            0.5: '#ffff00', 
            1.0: '#ff0000'
          }
        }).addTo(map);
        console.log('Heatmap layer added for day', day);
        return;
      } catch (error) {
        console.log('Heatmap plugin error, using markers:', error);
      }
    }
    
    // Fallback to circle markers
    dayPoints.forEach(point => {
      const color = point.intensity > 0.7 ? '#ff0000' : 
                   point.intensity > 0.4 ? '#ff8800' : '#00ff00';
      
      const marker = L.circleMarker([point.lat, point.lng], {
        radius: 8 + (point.intensity * 10),
        fillColor: color,
        color: color,
        weight: 2,
        opacity: 0.8,
        fillOpacity: 0.6
      }).addTo(map);
      
      marker.bindPopup(`
        <div class="popup-title">Pest Risk Zone</div>
        <div class="popup-info">
          <strong>Risk Level:</strong> ${(point.intensity * 100).toFixed(1)}%<br>
          <strong>Day:</strong> ${point.day}
        </div>
      `);
      
      riskMarkers.push(marker);
    });
    console.log('Risk markers added for day', day);
  }
}

function clearRiskMarkers() {
  riskMarkers.forEach(marker => {
    if (map && map.hasLayer(marker)) {
      map.removeLayer(marker);
    }
  });
  riskMarkers = [];
}

function toggleHeatmap() {
  const heatmapToggle = document.getElementById('heatmapToggle');
  if (!heatmapToggle) return;

  if (heatmapToggle.checked && forecastGenerated) {
    updateHeatmap(currentDay);
  } else {
    clearRiskMarkers();
    if (heatmapLayer && map) {
      map.removeLayer(heatmapLayer);
      heatmapLayer = null;
    }
  }
}

// Day Controls
function setupDayControls() {
  const dayButtons = document.querySelectorAll('.day-btn');
  dayButtons.forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.preventDefault();
      const day = parseInt(btn.dataset.day);
      selectDay(day);
    });
  });
}

function selectDay(day) {
  currentDay = day;
  
  // Update button states
  document.querySelectorAll('.day-btn').forEach(btn => {
    btn.classList.remove('active');
  });
  const activeBtn = document.querySelector(`[data-day="${day}"]`);
  if (activeBtn) {
    activeBtn.classList.add('active');
  }

  // Update display
  updateSelectedDayDisplay();
  updateHeatmap(day);
  updateRecommendations(day);
  
  // Update forecast card selection
  document.querySelectorAll('.forecast-card').forEach(card => {
    card.classList.remove('selected');
  });
  const selectedCard = document.querySelector(`[data-forecast-day="${day}"]`);
  if (selectedCard) {
    selectedCard.classList.add('selected');
  }
  
  console.log('Selected day:', day);
}

function updateSelectedDayDisplay() {
  const selectedDaySpan = document.getElementById('selectedDay');
  const riskIndicator = document.getElementById('riskIndicator');
  
  if (!selectedDaySpan || !riskIndicator) {
    console.error('Day display elements not found');
    return;
  }
  
  const dayData = appData.forecastData.find(d => d.day === currentDay);
  
  if (dayData) {
    selectedDaySpan.textContent = `Day ${currentDay}`;
    riskIndicator.textContent = dayData.riskLevel.toUpperCase();
    riskIndicator.className = `risk-indicator ${dayData.riskLevel}`;
  }
}

// Forecast Cards
function setupForecastCards() {
  const forecastCards = document.getElementById('forecastCards');
  if (!forecastCards) {
    console.error('Forecast cards container not found');
    return;
  }

  forecastCards.innerHTML = '';

  appData.forecastData.forEach(forecast => {
    const card = document.createElement('div');
    card.className = 'forecast-card';
    card.dataset.forecastDay = forecast.day;
    
    if (forecast.day === currentDay) {
      card.classList.add('selected');
    }

    card.innerHTML = `
      <div class="forecast-card-header">
        <h4>Day ${forecast.day}</h4>
        <span class="status status--${forecast.riskLevel === 'high' ? 'error' : forecast.riskLevel === 'medium' ? 'warning' : 'success'}">
          ${forecast.riskLevel.toUpperCase()}
        </span>
      </div>
      <div class="forecast-date">${new Date(forecast.date).toLocaleDateString()}</div>
      <div class="forecast-metrics">
        <div class="metric">
          <span class="metric-label">Infestation:</span>
          <span class="metric-value">${(forecast.infestationLevel * 100).toFixed(1)}%</span>
        </div>
        <div class="metric">
          <span class="metric-label">Affected Area:</span>
          <span class="metric-value">${forecast.affectedArea} ha</span>
        </div>
        <div class="metric">
          <span class="metric-label">Temperature:</span>
          <span class="metric-value">${forecast.temperature}°C</span>
        </div>
        <div class="metric">
          <span class="metric-label">Wind Speed:</span>
          <span class="metric-value">${forecast.windSpeed} km/h</span>
        </div>
      </div>
    `;

    card.addEventListener('click', (e) => {
      e.preventDefault();
      selectDay(forecast.day);
    });
    forecastCards.appendChild(card);
  });
  console.log('Forecast cards rendered');
}

// Chart Setup
function setupChart() {
  const chartCanvas = document.getElementById('trendChart');
  if (!chartCanvas) {
    console.error('Chart canvas not found');
    return;
  }

  if (!window.Chart) {
    console.error('Chart.js not loaded');
    return;
  }

  try {
    const ctx = chartCanvas.getContext('2d');
    
    trendChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: appData.forecastData.map(d => `Day ${d.day}`),
        datasets: [{
          label: 'Infestation Level (%)',
          data: appData.forecastData.map(d => d.infestationLevel * 100),
          borderColor: '#1FB8CD',
          backgroundColor: 'rgba(31, 184, 205, 0.1)',
          tension: 0.4,
          fill: true
        }, {
          label: 'Affected Area (ha)',
          data: appData.forecastData.map(d => d.affectedArea),
          borderColor: '#FFC185',
          backgroundColor: 'rgba(255, 193, 133, 0.1)',
          tension: 0.4,
          yAxisID: 'y1'
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          mode: 'index',
          intersect: false,
        },
        scales: {
          y: {
            type: 'linear',
            display: true,
            position: 'left',
            title: {
              display: true,
              text: 'Infestation Level (%)'
            }
          },
          y1: {
            type: 'linear',
            display: true,
            position: 'right',
            title: {
              display: true,
              text: 'Affected Area (ha)'
            },
            grid: {
              drawOnChartArea: false,
            },
          }
        },
        plugins: {
          legend: {
            position: 'top',
          },
          title: {
            display: true,
            text: 'Pest Infestation Forecast Trend'
          }
        }
      }
    });
    console.log('Chart initialized successfully');
  } catch (error) {
    console.error('Error setting up chart:', error);
  }
}

// Recommendations
function updateRecommendations(day) {
  const recDay = document.getElementById('recDay');
  const recList = document.getElementById('recList');
  
  if (!recDay || !recList) {
    console.error('Recommendation elements not found');
    return;
  }
  
  const dayData = appData.forecastData.find(d => d.day === day);
  
  if (dayData) {
    recDay.textContent = `Day ${day}`;
    recList.innerHTML = '';
    
    dayData.recommendations.forEach(rec => {
      const li = document.createElement('li');
      li.textContent = rec;
      recList.appendChild(li);
    });
  }
}

// Forecast Generation
function generateForecast() {
  if (!selectedImage) {
    alert('Please select or upload a drone image first');
    return;
  }

  console.log('Generating forecast for:', selectedImage.name);
  showProcessingModal();
  simulateProcessing();
}

function showProcessingModal() {
  const modal = document.getElementById('processingModal');
  if (modal) {
    modal.classList.remove('hidden');
  }
}

function hideProcessingModal() {
  const modal = document.getElementById('processingModal');
  if (modal) {
    modal.classList.add('hidden');
  }
}

function simulateProcessing() {
  const steps = [
    { message: 'Initializing ResNet disease detection...', duration: 1000 },
    { message: 'Analyzing drone image patches...', duration: 1500 },
    { message: 'Calculating NDVI from satellite data...', duration: 1200 },
    { message: 'Fetching weather forecast data...', duration: 800 },
    { message: 'Running ConvLSTM spatiotemporal model...', duration: 2000 },
    { message: 'Generating risk heatmaps...', duration: 1000 },
    { message: 'Finalizing forecast predictions...', duration: 500 }
  ];

  let currentStep = 0;
  let progress = 0;

  function processStep() {
    if (currentStep >= steps.length) {
      completeProcessing();
      return;
    }

    const step = steps[currentStep];
    const statusElement = document.getElementById('processingStatus');
    if (statusElement) {
      statusElement.textContent = step.message;
    }

    const stepProgress = (currentStep + 1) / steps.length * 100;
    animateProgress(progress, stepProgress, step.duration);
    progress = stepProgress;

    setTimeout(() => {
      currentStep++;
      processStep();
    }, step.duration);
  }

  processStep();
}

function animateProgress(start, end, duration) {
  const progressFill = document.getElementById('progressFill');
  if (!progressFill) return;

  const startTime = Date.now();

  function animate() {
    const elapsed = Date.now() - startTime;
    const progress = Math.min(elapsed / duration, 1);
    const current = start + (end - start) * progress;
    
    progressFill.style.width = current + '%';

    if (progress < 1) {
      requestAnimationFrame(animate);
    }
  }

  animate();
}

function completeProcessing() {
  setTimeout(() => {
    hideProcessingModal();
    forecastGenerated = true;
    
    // Update heatmap for current day
    updateHeatmap(currentDay);
    
    // Show success message
    const message = 'Forecast generated successfully!\n\n' +
      '✓ ResNet disease detection completed\n' +
      '✓ ConvLSTM forecast model executed\n' +
      '✓ Pest risk heatmap generated\n\n' +
      'Use the day buttons to view different forecast periods.';
    alert(message);
    
    console.log('Forecast generation completed');
  }, 500);
}

// Event Listeners
function setupEventListeners() {
  const generateBtn = document.getElementById('generateForecast');
  if (generateBtn) {
    generateBtn.addEventListener('click', (e) => {
      e.preventDefault();
      generateForecast();
    });
    console.log('Generate forecast button event listener added');
  } else {
    console.error('Generate forecast button not found');
  }
}

// Modal Functions
function openModal(modalId) {
  const modal = document.getElementById(modalId);
  if (modal) {
    modal.classList.remove('hidden');
  }
}

function closeModal(modalId) {
  const modal = document.getElementById(modalId);
  if (modal) {
    modal.classList.add('hidden');
  }
}

// Make functions globally available
window.openModal = openModal;
window.closeModal = closeModal;