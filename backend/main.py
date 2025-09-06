
"""
Phase 4: FastAPI Backend for Pest Forecasting System
Handles file uploads, runs ML pipeline, and serves forecast data
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import uvicorn
import os
import json
import numpy as np
from datetime import datetime, timedelta
import logging

# Import our custom modules
import sys
sys.path.append('backend')
from utils.satellite_processor import SatelliteDataProcessor
from utils.weather_processor import WeatherDataProcessor  
from models.plant_disease_detector import PlantDiseaseDetector
from models.convlstm_forecaster import PestSpreadForecaster

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Hyper-Local Pest Infestation Forecaster", version="1.0.0")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "*"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global instances
satellite_processor = SatelliteDataProcessor()
weather_processor = WeatherDataProcessor()
disease_detector = PlantDiseaseDetector()
forecaster = PestSpreadForecaster()

# Pydantic models
class FarmLocation(BaseModel):
    latitude: float
    longitude: float
    bbox: list[float]  # [min_lon, min_lat, max_lon, max_lat]

class ForecastRequest(BaseModel):
    farm_location: FarmLocation
    forecast_days: int = 3

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    logger.info("Initializing ML models...")

    # Load or train disease detection model
    try:
        disease_detector.load_model()
    except:
        logger.info("Training disease detection model...")
        disease_detector.create_synthetic_training_data(num_samples=400)
        disease_detector.build_resnet_model()
        disease_detector.train_model(epochs=3)

    # Load or train forecasting model
    try:
        forecaster.load_model()
    except:
        logger.info("Training forecasting model...")
        X_train, y_train = forecaster.create_synthetic_pest_spread_data(
            num_sequences=200, sequence_length=10, grid_size=64
        )
        forecaster.build_convlstm_model()
        forecaster.train_model(X_train, y_train, epochs=10, batch_size=4)

    logger.info("ML models initialized successfully")

@app.get("/")
async def root():
    """API health check"""
    return {"message": "Pest Forecasting API is running", "timestamp": datetime.now()}

@app.post("/upload-drone-image")
async def upload_drone_image(file: UploadFile = File(...)):
    """
    Upload drone image and generate initial pest map
    """
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Save uploaded file
        upload_dir = "backend/data/uploads"
        os.makedirs(upload_dir, exist_ok=True)

        file_path = os.path.join(upload_dir, file.filename)

        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        logger.info(f"Uploaded file: {file.filename}")

        # Process image to create pest map
        pest_map_path = "backend/data/geotiff/initial_pests.tif"

        pest_map = disease_detector.create_pest_map_geotiff(
            file_path, 
            pest_map_path,
            bbox=[-74.2, 40.7, -74.0, 40.9]  # Default bbox, should be from request
        )

        # Calculate statistics
        if pest_map is not None:
            stats = {
                "total_area_km2": 4.0,  # Example area
                "infested_percentage": float(np.mean(pest_map > 0.3) * 100),
                "max_infestation_level": float(np.max(pest_map)),
                "avg_infestation_level": float(np.mean(pest_map[pest_map > 0]))
            }
        else:
            stats = {"error": "Failed to process image"}

        return JSONResponse({
            "status": "success",
            "filename": file.filename,
            "pest_map_path": pest_map_path,
            "statistics": stats
        })

    except Exception as e:
        logger.error(f"Error processing upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-forecast")
async def generate_forecast(request: ForecastRequest):
    """
    Generate pest spread forecast for specified farm location
    """
    try:
        logger.info(f"Generating forecast for location: {request.farm_location.latitude}, {request.farm_location.longitude}")

        # Step 1: Get satellite data
        satellite_processor.download_sentinel2_data(
            request.farm_location.bbox,
            (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
            datetime.now().strftime("%Y-%m-%d")
        )

        # Step 2: Calculate NDVI
        red_path = "backend/data/satellite/DEMO_S2A_MSIL1C_B04_red.tif"
        nir_path = "backend/data/satellite/DEMO_S2A_MSIL1C_B08_nir.tif"
        ndvi_path = "backend/data/geotiff/ndvi.tif"

        satellite_processor.calculate_ndvi(red_path, nir_path, ndvi_path)

        # Step 3: Get weather data
        weather_processor.get_weather_forecast(
            request.farm_location.latitude,
            request.farm_location.longitude,
            days=request.forecast_days
        )

        # Step 4: Create weather GeoTIFFs
        hourly_df, daily_df = weather_processor.get_weather_forecast(
            request.farm_location.latitude, request.farm_location.longitude
        )

        weather_processor.create_weather_geotiff(
            hourly_df, 'temperature_2m', request.farm_location.bbox,
            output_path="backend/data/geotiff/temperature.tif"
        )

        weather_processor.create_weather_geotiff(
            hourly_df, 'wind_speed_10m', request.farm_location.bbox,
            output_path="backend/data/geotiff/wind_speed.tif"
        )

        # Step 5: Generate forecasts using ConvLSTM
        geotiff_files = {
            'pests': 'backend/data/geotiff/initial_pests.tif',
            'ndvi': 'backend/data/geotiff/ndvi.tif',
            'temperature': 'backend/data/geotiff/temperature.tif',
            'wind': 'backend/data/geotiff/wind_speed.tif'
        }

        forecasts = forecaster.create_forecast_stack(geotiff_files)

        # Prepare response data
        forecast_data = []
        for day in range(request.forecast_days):
            if forecasts and day < len(forecasts):
                forecast_stats = {
                    "day": day + 1,
                    "date": (datetime.now() + timedelta(days=day+1)).strftime("%Y-%m-%d"),
                    "infestation_level": float(np.mean(forecasts[day])),
                    "risk_level": "high" if np.mean(forecasts[day]) > 0.6 else "medium" if np.mean(forecasts[day]) > 0.3 else "low",
                    "affected_area_percentage": float(np.mean(forecasts[day] > 0.3) * 100),
                    "geotiff_path": f"backend/data/geotiff/forecasts/pest_forecast_day_{day+1}.tif"
                }
            else:
                forecast_stats = {
                    "day": day + 1,
                    "date": (datetime.now() + timedelta(days=day+1)).strftime("%Y-%m-%d"),
                    "infestation_level": 0.2,  # Default values
                    "risk_level": "low",
                    "affected_area_percentage": 5.0,
                    "geotiff_path": ""
                }

            forecast_data.append(forecast_stats)

        return JSONResponse({
            "status": "success",
            "forecast_generated_at": datetime.now().isoformat(),
            "farm_location": request.farm_location.dict(),
            "forecasts": forecast_data,
            "recommendations": [
                "Monitor areas with high infestation probability",
                "Consider preventive treatment in forecasted hotspots", 
                "Increase surveillance in the next 48 hours"
            ]
        })

    except Exception as e:
        logger.error(f"Error generating forecast: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/forecast-data/{day}")
async def get_forecast_data(day: int):
    """
    Get forecast data for specific day as JSON (for heatmap visualization)
    """
    try:
        geotiff_path = f"backend/data/geotiff/forecasts/pest_forecast_day_{day}.tif"

        if os.path.exists(geotiff_path):
            import rasterio

            with rasterio.open(geotiff_path) as src:
                data = src.read(1)
                transform = src.transform
                bounds = src.bounds

            # Convert to coordinate points for heatmap
            height, width = data.shape
            points = []

            # Sample every nth pixel to reduce data size
            step = max(1, max(height, width) // 50)  # Max 50x50 grid

            for i in range(0, height, step):
                for j in range(0, width, step):
                    if data[i, j] > 0.1:  # Only include significant values
                        # Convert pixel coordinates to lat/lon
                        x, y = rasterio.transform.xy(transform, i, j)
                        points.append({
                            "lat": y,
                            "lng": x, 
                            "intensity": float(data[i, j])
                        })

            return JSONResponse({
                "day": day,
                "bounds": {
                    "north": bounds.top,
                    "south": bounds.bottom,
                    "east": bounds.right,
                    "west": bounds.left
                },
                "points": points[:1000]  # Limit points for performance
            })
        else:
            # Return demo data
            return JSONResponse({
                "day": day,
                "bounds": {
                    "north": 40.9,
                    "south": 40.7,
                    "east": -74.0,
                    "west": -74.2
                },
                "points": [
                    {"lat": 40.8, "lng": -74.1, "intensity": 0.7},
                    {"lat": 40.82, "lng": -74.12, "intensity": 0.5},
                    {"lat": 40.85, "lng": -74.08, "intensity": 0.9}
                ]
            })

    except Exception as e:
        logger.error(f"Error getting forecast data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download-geotiff/{filename}")
async def download_geotiff(filename: str):
    """Download GeoTIFF files"""
    file_path = f"backend/data/geotiff/{filename}"

    if os.path.exists(file_path):
        return FileResponse(
            file_path,
            media_type="application/octet-stream",
            filename=filename
        )
    else:
        raise HTTPException(status_code=404, detail="File not found")

@app.get("/system-status")
async def system_status():
    """Get system status and model information"""
    return JSONResponse({
        "status": "operational",
        "models_loaded": {
            "disease_detector": disease_detector.model is not None,
            "forecaster": forecaster.model is not None
        },
        "data_sources": {
            "satellite": "Sentinel-2 (Copernicus)",
            "weather": "Open-Meteo API"
        },
        "last_updated": datetime.now().isoformat()
    })

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0", 
        port=8000,
        reload=True
    )
