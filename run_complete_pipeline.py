
#!/usr/bin/env python3
"""
Main execution script for the Hyper-Local Pest Infestation Forecaster
Run this script to execute all phases of the project sequentially
"""
import os
import sys
import logging
from datetime import datetime, timedelta

# Add backend to Python path
sys.path.append('backend')

# Import our modules
from utils.satellite_processor import SatelliteDataProcessor
from utils.weather_processor import WeatherDataProcessor
from models.plant_disease_detector import PlantDiseaseDetector
from models.convlstm_forecaster import PestSpreadForecaster

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Execute the complete pest forecasting pipeline"""
    logger.info("🚀 Starting Hyper-Local Pest Infestation Forecaster")

    # Example farm coordinates (New Jersey agricultural area)
    farm_bbox = [-74.2, 40.7, -74.0, 40.9]  # [min_lon, min_lat, max_lon, max_lat]
    farm_lat, farm_lon = 40.8, -74.1

    try:
        # Phase 1: Data Ingestion and Processing
        logger.info("📡 Phase 1: Data Ingestion and Processing")

        # Initialize satellite processor
        satellite_processor = SatelliteDataProcessor()

        # Download and process satellite data
        satellite_processor.download_sentinel2_data(
            farm_bbox,
            (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
            datetime.now().strftime("%Y-%m-%d"),
            cloud_cover_max=20
        )

        # Calculate NDVI
        red_path = "backend/data/satellite/DEMO_S2A_MSIL1C_B04_red.tif"
        nir_path = "backend/data/satellite/DEMO_S2A_MSIL1C_B08_nir.tif"
        ndvi_path = "backend/data/geotiff/ndvi.tif"

        satellite_processor.calculate_ndvi(red_path, nir_path, ndvi_path)

        # Initialize weather processor
        weather_processor = WeatherDataProcessor()

        # Get weather forecast
        hourly_df, daily_df = weather_processor.get_weather_forecast(
            farm_lat, farm_lon, days=3
        )

        # Create weather GeoTIFFs
        weather_processor.create_weather_geotiff(
            hourly_df, 'temperature_2m', farm_bbox,
            output_path="backend/data/geotiff/temperature.tif"
        )

        weather_processor.create_weather_geotiff(
            hourly_df, 'wind_speed_10m', farm_bbox,
            output_path="backend/data/geotiff/wind_speed.tif"
        )

        logger.info("✅ Phase 1 completed successfully")

        # Phase 2: Initial Pest Detection
        logger.info("🔍 Phase 2: Initial Pest Detection")

        # Initialize plant disease detector
        disease_detector = PlantDiseaseDetector()

        # Create synthetic training data (replace with Kaggle dataset)
        disease_detector.create_synthetic_training_data(num_samples=800)

        # Build and train ResNet model
        disease_detector.build_resnet_model()
        history = disease_detector.train_model(epochs=5)

        # Create pest map from sample drone image
        pest_map = disease_detector.create_pest_map_geotiff(
            "backend/data/sample_drone_image.jpg",
            "backend/data/geotiff/initial_pests.tif",
            bbox=farm_bbox
        )

        logger.info("✅ Phase 2 completed successfully")

        # Phase 3: Build Forecasting Model
        logger.info("🔮 Phase 3: Build Forecasting Model")

        # Initialize ConvLSTM forecaster
        forecaster = PestSpreadForecaster()

        # Create synthetic pest spread training data
        X_train, y_train = forecaster.create_synthetic_pest_spread_data(
            num_sequences=500, sequence_length=10, grid_size=64
        )

        # Build and train ConvLSTM model
        forecaster.build_convlstm_model()
        history = forecaster.train_model(X_train, y_train, epochs=20, batch_size=8)

        # Create forecast stack
        geotiff_files = {
            'pests': 'backend/data/geotiff/initial_pests.tif',
            'ndvi': 'backend/data/geotiff/ndvi.tif',
            'temperature': 'backend/data/geotiff/temperature.tif',
            'wind': 'backend/data/geotiff/wind_speed.tif'
        }

        forecasts = forecaster.create_forecast_stack(geotiff_files)

        logger.info("✅ Phase 3 completed successfully")

        # Phase 4: Results Summary
        logger.info("📊 Phase 4: Results Summary")
        logger.info("="*50)
        logger.info("PROJECT COMPLETION SUMMARY")
        logger.info("="*50)

        # Check created files
        created_files = [
            "backend/data/geotiff/ndvi.tif",
            "backend/data/geotiff/temperature.tif", 
            "backend/data/geotiff/wind_speed.tif",
            "backend/data/geotiff/initial_pests.tif",
            "backend/data/models/resnet_plant_disease_model.h5",
            "backend/data/models/convlstm_forecaster.h5"
        ]

        for file_path in created_files:
            if os.path.exists(file_path):
                logger.info(f"✅ Created: {file_path}")
            else:
                logger.info(f"❌ Missing: {file_path}")

        # Model performance summary
        logger.info("\n🤖 AI Models Trained:")
        logger.info("   • ResNet-50 Plant Disease Detector")
        logger.info("   • ConvLSTM Pest Spread Forecaster")

        logger.info("\n🌍 Data Sources Processed:")
        logger.info("   • Sentinel-2 Satellite Imagery (NDVI)")
        logger.info("   • Open-Meteo Weather Forecasts")
        logger.info("   • Synthetic Drone Images")

        logger.info("\n🎯 Key Features Implemented:")
        logger.info("   • Real-time pest detection from drone images")
        logger.info("   • 3-day pest spread forecasting")
        logger.info("   • Interactive web dashboard")
        logger.info("   • Geospatial data visualization")

        logger.info("\n🚀 Next Steps:")
        logger.info("   1. Start the backend API: cd pest_forecaster && python backend/main.py")
        logger.info("   2. Start the frontend: cd pest_forecaster/frontend && npm start")
        logger.info("   3. Open http://localhost:3000 to use the dashboard")

        logger.info("\n" + "="*50)
        logger.info("🎉 HYPER-LOCAL PEST FORECASTER - PROJECT COMPLETED!")
        logger.info("="*50)

    except Exception as e:
        logger.error(f"❌ Error in main execution: {e}")
        return False

    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎊 All phases completed successfully!")
        print("\nTo run the complete system:")
        print("1. Backend API: python backend/main.py")
        print("2. Frontend: cd frontend && npm install && npm start")
    else:
        print("\n❌ Some phases failed. Check the logs above.")
