
"""
Phase 1: Satellite Data Ingestion using Sentinel-2
Download and process Sentinel-2 satellite imagery
"""
import os
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.enums import Resampling
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SatelliteDataProcessor:
    def __init__(self, username="demo", password="demo"):
        """
        Initialize Satellite Data Processor
        Note: For production, use real credentials from Copernicus Hub
        """
        self.api = SentinelAPI(username, password, 'https://apihub.copernicus.eu/apihub')

    def download_sentinel2_data(self, bbox, start_date, end_date, cloud_cover_max=30):
        """
        Download Sentinel-2 data for specified area and time range

        Args:
            bbox: Bounding box [min_lon, min_lat, max_lon, max_lat]
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)  
            cloud_cover_max: Maximum cloud cover percentage
        """
        try:
            # Create footprint from bounding box
            footprint_coords = [
                [bbox[0], bbox[1]], [bbox[2], bbox[1]], 
                [bbox[2], bbox[3]], [bbox[0], bbox[3]], 
                [bbox[0], bbox[1]]
            ]

            footprint = f"POLYGON(({','.join([f'{lon} {lat}' for lon, lat in footprint_coords])}))"

            # Query Sentinel-2 products
            products = self.api.query(
                footprint,
                date=(start_date, end_date),
                platformname='Sentinel-2',
                cloudcoverpercentage=(0, cloud_cover_max),
                producttype='S2MSI1C'
            )

            logger.info(f"Found {len(products)} products")

            # Download products (for demo, we'll simulate this)
            for product_id, product_info in list(products.items())[:1]:  # Download first product only
                logger.info(f"Processing product: {product_info['title']}")
                # In production: self.api.download(product_id, directory_path='./data/satellite/')

                # For this demo, we'll create a simulated product
                self._create_simulated_sentinel2_data(product_info['title'])

        except Exception as e:
            logger.error(f"Error downloading satellite data: {e}")
            # Create simulated data for demo
            self._create_simulated_sentinel2_data("DEMO_S2A_MSIL1C")

    def _create_simulated_sentinel2_data(self, product_name):
        """Create simulated Sentinel-2 data for demonstration"""
        logger.info("Creating simulated Sentinel-2 data for demo...")

        # Simulate different bands (Red, NIR, etc.)
        height, width = 1000, 1000

        # Create realistic-looking agricultural data
        np.random.seed(42)

        # Red band (Band 4) - typically lower values for vegetation
        red_band = np.random.normal(0.15, 0.05, (height, width)).clip(0, 1)

        # NIR band (Band 8) - typically higher values for vegetation  
        nir_band = np.random.normal(0.45, 0.15, (height, width)).clip(0, 1)

        # Add some spatial patterns to simulate fields
        y, x = np.ogrid[:height, :width]

        # Simulate agricultural field patterns
        field_pattern1 = np.sin(y/100) * np.sin(x/100) * 0.1
        field_pattern2 = np.sin(y/50) * np.sin(x/150) * 0.08

        red_band += field_pattern1
        nir_band += field_pattern2

        red_band = red_band.clip(0, 1)
        nir_band = nir_band.clip(0, 1)

        # Define geospatial transform (example coordinates)
        transform = rasterio.transform.from_bounds(
            -74.2, 40.7, -74.0, 40.9, width, height
        )

        # Save Red band
        self._save_geotiff(
            red_band, 
            f"backend/data/satellite/{product_name}_B04_red.tif", 
            transform
        )

        # Save NIR band  
        self._save_geotiff(
            nir_band,
            f"backend/data/satellite/{product_name}_B08_nir.tif",
            transform
        )

        logger.info("Simulated Sentinel-2 data created successfully")

    def _save_geotiff(self, data, filepath, transform):
        """Save numpy array as GeoTIFF"""
        profile = {
            'driver': 'GTiff',
            'dtype': rasterio.float32,
            'nodata': None,
            'width': data.shape[1],
            'height': data.shape[0],
            'count': 1,
            'crs': 'EPSG:4326',
            'transform': transform,
            'compress': 'lzw'
        }

        with rasterio.open(filepath, 'w', **profile) as dst:
            dst.write(data.astype(rasterio.float32), 1)

    def calculate_ndvi(self, red_band_path, nir_band_path, output_path):
        """
        Calculate NDVI from Red and NIR bands
        NDVI = (NIR - Red) / (NIR + Red)
        """
        try:
            with rasterio.open(red_band_path) as red_src:
                red_data = red_src.read(1).astype(np.float32)
                profile = red_src.profile

            with rasterio.open(nir_band_path) as nir_src:
                nir_data = nir_src.read(1).astype(np.float32)

            # Calculate NDVI with safe division
            numerator = nir_data - red_data
            denominator = nir_data + red_data

            # Avoid division by zero
            ndvi = np.where(denominator != 0, numerator / denominator, 0)
            ndvi = np.clip(ndvi, -1, 1)  # NDVI should be between -1 and 1

            # Save NDVI as GeoTIFF
            profile.update(dtype=rasterio.float32, nodata=-9999)

            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(ndvi, 1)

            logger.info(f"NDVI calculated and saved to {output_path}")
            return ndvi

        except Exception as e:
            logger.error(f"Error calculating NDVI: {e}")
            return None

if __name__ == "__main__":
    # Example usage
    processor = SatelliteDataProcessor()

    # Example farm coordinates (New Jersey area)
    farm_bbox = [-74.2, 40.7, -74.0, 40.9]  # [min_lon, min_lat, max_lon, max_lat]

    # Download satellite data
    processor.download_sentinel2_data(
        farm_bbox, 
        "2023-06-01", 
        "2023-06-30", 
        cloud_cover_max=20
    )

    # Calculate NDVI
    red_path = "backend/data/satellite/DEMO_S2A_MSIL1C_B04_red.tif"
    nir_path = "backend/data/satellite/DEMO_S2A_MSIL1C_B08_nir.tif"
    ndvi_path = "backend/data/geotiff/ndvi.tif"

    processor.calculate_ndvi(red_path, nir_path, ndvi_path)
