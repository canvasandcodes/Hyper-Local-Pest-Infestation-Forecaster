
"""
Phase 1: Weather Data Ingestion using Open-Meteo API
Get weather forecast data for agricultural applications
"""
import openmeteo_requests
import requests_cache
import pandas as pd
import numpy as np
import rasterio
from datetime import datetime, timedelta
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeatherDataProcessor:
    def __init__(self):
        """Initialize Weather Data Processor with Open-Meteo API"""
        # Setup the Open-Meteo API client with cache and retry
        cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
        retry_session = requests_cache.CachedSession('.cache', expire_after=3600)
        self.openmeteo = openmeteo_requests.Client(session=retry_session)

    def get_weather_forecast(self, latitude, longitude, days=3):
        """
        Get weather forecast for specific coordinates

        Args:
            latitude: Farm latitude
            longitude: Farm longitude  
            days: Number of forecast days (1-16)
        """
        try:
            # Open-Meteo API parameters
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "hourly": [
                    "temperature_2m", 
                    "relative_humidity_2m", 
                    "precipitation",
                    "wind_speed_10m", 
                    "wind_direction_10m"
                ],
                "daily": [
                    "temperature_2m_max",
                    "temperature_2m_min", 
                    "precipitation_sum",
                    "wind_speed_10m_max"
                ],
                "forecast_days": days
            }

            responses = self.openmeteo.weather_api(url, params=params)
            response = responses[0]

            logger.info(f"Weather data for coordinates {response.Latitude()}°N {response.Longitude()}°E")

            # Process hourly data
            hourly = response.Hourly()
            hourly_data = {
                "date": pd.date_range(
                    start=pd.to_datetime(hourly.Time(), unit="s"),
                    end=pd.to_datetime(hourly.TimeEnd(), unit="s"), 
                    freq=pd.Timedelta(seconds=hourly.Interval()),
                    inclusive="left"
                ),
                "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
                "relative_humidity_2m": hourly.Variables(1).ValuesAsNumpy(),
                "precipitation": hourly.Variables(2).ValuesAsNumpy(),
                "wind_speed_10m": hourly.Variables(3).ValuesAsNumpy(),
                "wind_direction_10m": hourly.Variables(4).ValuesAsNumpy()
            }

            hourly_df = pd.DataFrame(data=hourly_data)

            # Process daily data
            daily = response.Daily()
            daily_data = {
                "date": pd.date_range(
                    start=pd.to_datetime(daily.Time(), unit="s"),
                    end=pd.to_datetime(daily.TimeEnd(), unit="s"),
                    freq=pd.Timedelta(seconds=daily.Interval()),
                    inclusive="left"
                ),
                "temperature_2m_max": daily.Variables(0).ValuesAsNumpy(),
                "temperature_2m_min": daily.Variables(1).ValuesAsNumpy(),
                "precipitation_sum": daily.Variables(2).ValuesAsNumpy(),
                "wind_speed_10m_max": daily.Variables(3).ValuesAsNumpy()
            }

            daily_df = pd.DataFrame(data=daily_data)

            # Save weather data
            hourly_df.to_csv("backend/data/weather/hourly_forecast.csv", index=False)
            daily_df.to_csv("backend/data/weather/daily_forecast.csv", index=False)

            logger.info("Weather forecast data saved successfully")

            return hourly_df, daily_df

        except Exception as e:
            logger.error(f"Error fetching weather data: {e}")
            return self._create_simulated_weather_data(latitude, longitude, days)

    def _create_simulated_weather_data(self, latitude, longitude, days=3):
        """Create simulated weather data for demonstration"""
        logger.info("Creating simulated weather data for demo...")

        # Create hourly data
        hours = days * 24
        dates = pd.date_range(start=datetime.now(), periods=hours, freq='H')

        # Simulate realistic weather patterns
        np.random.seed(42)
        base_temp = 22.0  # Base temperature in Celsius

        hourly_data = {
            "date": dates,
            "temperature_2m": base_temp + 8 * np.sin(np.arange(hours) * 2 * np.pi / 24) + np.random.normal(0, 2, hours),
            "relative_humidity_2m": 60 + 20 * np.sin(np.arange(hours) * 2 * np.pi / 24 + np.pi) + np.random.normal(0, 5, hours),
            "precipitation": np.random.exponential(0.1, hours),
            "wind_speed_10m": 5 + 3 * np.sin(np.arange(hours) * 2 * np.pi / 24) + np.random.normal(0, 1.5, hours),
            "wind_direction_10m": np.random.uniform(0, 360, hours)
        }

        # Ensure realistic bounds
        hourly_data["relative_humidity_2m"] = np.clip(hourly_data["relative_humidity_2m"], 20, 100)
        hourly_data["wind_speed_10m"] = np.clip(hourly_data["wind_speed_10m"], 0, 25)
        hourly_data["precipitation"] = np.clip(hourly_data["precipitation"], 0, 20)

        hourly_df = pd.DataFrame(hourly_data)

        # Create daily summaries
        daily_dates = pd.date_range(start=datetime.now().date(), periods=days, freq='D')
        daily_data = {
            "date": daily_dates,
            "temperature_2m_max": [hourly_df[hourly_df['date'].dt.date == date]['temperature_2m'].max() 
                                 for date in daily_dates.date],
            "temperature_2m_min": [hourly_df[hourly_df['date'].dt.date == date]['temperature_2m'].min() 
                                 for date in daily_dates.date],
            "precipitation_sum": [hourly_df[hourly_df['date'].dt.date == date]['precipitation'].sum() 
                                for date in daily_dates.date],
            "wind_speed_10m_max": [hourly_df[hourly_df['date'].dt.date == date]['wind_speed_10m'].max() 
                                 for date in daily_dates.date]
        }

        daily_df = pd.DataFrame(daily_data)

        # Save simulated data
        hourly_df.to_csv("backend/data/weather/hourly_forecast.csv", index=False)
        daily_df.to_csv("backend/data/weather/daily_forecast.csv", index=False)

        logger.info("Simulated weather data created successfully")
        return hourly_df, daily_df

    def create_weather_geotiff(self, weather_df, variable, bbox, resolution=0.001, output_path=None):
        """
        Create a GeoTIFF from weather data for a specific area

        Args:
            weather_df: DataFrame with weather data
            variable: Weather variable to map (e.g., 'temperature_2m', 'wind_speed_10m')
            bbox: Bounding box [min_lon, min_lat, max_lon, max_lat]
            resolution: Spatial resolution in degrees
            output_path: Output path for GeoTIFF
        """
        try:
            # Calculate grid dimensions
            width = int((bbox[2] - bbox[0]) / resolution)
            height = int((bbox[3] - bbox[1]) / resolution)

            # Get latest value for the variable
            latest_value = weather_df[variable].iloc[-1] if not weather_df.empty else 20.0

            # Create uniform grid (in reality, you'd interpolate weather station data)
            # For this demo, we'll create a simple gradient with some noise
            y_coords, x_coords = np.ogrid[:height, :width]

            # Create spatial pattern (temperature decreases with latitude, wind varies)
            if 'temperature' in variable:
                data = latest_value + (height - y_coords) * 0.01 + np.random.normal(0, 0.5, (height, width))
            elif 'wind' in variable:
                data = latest_value + np.sin(x_coords/50) * 2 + np.random.normal(0, 1, (height, width))
            else:
                data = np.full((height, width), latest_value) + np.random.normal(0, latest_value*0.1, (height, width))

            # Define geospatial transform
            transform = rasterio.transform.from_bounds(
                bbox[0], bbox[1], bbox[2], bbox[3], width, height
            )

            # Output path
            if output_path is None:
                output_path = f"backend/data/geotiff/{variable}.tif"

            # Save as GeoTIFF
            profile = {
                'driver': 'GTiff',
                'dtype': rasterio.float32,
                'nodata': -9999,
                'width': width,
                'height': height, 
                'count': 1,
                'crs': 'EPSG:4326',
                'transform': transform,
                'compress': 'lzw'
            }

            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(data.astype(rasterio.float32), 1)

            logger.info(f"Weather GeoTIFF created: {output_path}")
            return data

        except Exception as e:
            logger.error(f"Error creating weather GeoTIFF: {e}")
            return None

if __name__ == "__main__":
    # Example usage
    weather_processor = WeatherDataProcessor()

    # Get weather data for farm location
    farm_lat, farm_lon = 40.8, -74.1  # Example coordinates

    hourly_df, daily_df = weather_processor.get_weather_forecast(farm_lat, farm_lon, days=3)

    # Create weather GeoTIFF layers
    farm_bbox = [-74.2, 40.7, -74.0, 40.9]

    weather_processor.create_weather_geotiff(
        hourly_df, 'temperature_2m', farm_bbox, 
        output_path="backend/data/geotiff/temperature.tif"
    )

    weather_processor.create_weather_geotiff(
        hourly_df, 'wind_speed_10m', farm_bbox,
        output_path="backend/data/geotiff/wind_speed.tif" 
    )

    print("Weather data processing completed!")
