
import React, { useState, useCallback } from 'react';
import MapComponent from './components/MapComponent';
import FileUpload from './components/FileUpload';
import ForecastPanel from './components/ForecastPanel';
import StatusPanel from './components/StatusPanel';
import './App.css';

export interface ForecastData {
  day: number;
  date: string;
  infestation_level: number;
  risk_level: 'low' | 'medium' | 'high';
  affected_area_percentage: number;
  geotiff_path: string;
}

export interface FarmLocation {
  latitude: number;
  longitude: number;
  bbox: [number, number, number, number]; // [min_lon, min_lat, max_lon, max_lat]
}

function App() {
  const [forecasts, setForecasts] = useState<ForecastData[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedDay, setSelectedDay] = useState(1);
  const [farmLocation, setFarmLocation] = useState<FarmLocation>({
    latitude: 40.8,
    longitude: -74.1,
    bbox: [-74.2, 40.7, -74.0, 40.9]
  });
  const [uploadedFile, setUploadedFile] = useState<string | null>(null);
  const [systemStatus, setSystemStatus] = useState<any>(null);

  const handleFileUpload = useCallback(async (file: File) => {
    setIsLoading(true);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('/upload-drone-image', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Upload failed');
      }

      const result = await response.json();
      setUploadedFile(result.filename);

      // Auto-generate forecast after successful upload
      await generateForecast();

    } catch (error) {
      console.error('Upload error:', error);
      alert('Failed to upload file. Please try again.');
    } finally {
      setIsLoading(false);
    }
  }, []);

  const generateForecast = useCallback(async () => {
    setIsLoading(true);

    try {
      const response = await fetch('/generate-forecast', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          farm_location: farmLocation,
          forecast_days: 3
        }),
      });

      if (!response.ok) {
        throw new Error('Forecast generation failed');
      }

      const result = await response.json();
      setForecasts(result.forecasts);

    } catch (error) {
      console.error('Forecast error:', error);
      alert('Failed to generate forecast. Please try again.');
    } finally {
      setIsLoading(false);
    }
  }, [farmLocation]);

  const fetchSystemStatus = useCallback(async () => {
    try {
      const response = await fetch('/system-status');
      if (response.ok) {
        const status = await response.json();
        setSystemStatus(status);
      }
    } catch (error) {
      console.error('Status fetch error:', error);
    }
  }, []);

  React.useEffect(() => {
    fetchSystemStatus();
  }, [fetchSystemStatus]);

  return (
    <div className="App">
      <header className="app-header">
        <h1>🌾 Hyper-Local Pest Infestation Forecaster</h1>
        <p>AI-powered pest spread prediction for precision agriculture</p>
      </header>

      <div className="app-container">
        <div className="left-panel">
          <FileUpload onFileUpload={handleFileUpload} isLoading={isLoading} />

          <ForecastPanel
            forecasts={forecasts}
            selectedDay={selectedDay}
            onDaySelect={setSelectedDay}
            onGenerateForecast={generateForecast}
            isLoading={isLoading}
          />

          <StatusPanel 
            systemStatus={systemStatus}
            uploadedFile={uploadedFile}
          />
        </div>

        <div className="map-container">
          <MapComponent
            farmLocation={farmLocation}
            selectedDay={selectedDay}
            forecasts={forecasts}
            isLoading={isLoading}
          />
        </div>
      </div>
    </div>
  );
}

export default App;
