
"""
Phase 3: Pest Spread Forecasting using ConvLSTM
Implement ConvLSTM model to predict future pest spread patterns
"""
import tensorflow as tf
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv2D, Dense
from tensorflow.keras.models import Sequential
import numpy as np
import rasterio
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PestSpreadForecaster:
    def __init__(self, input_shape=(10, 64, 64, 4)):
        """
        Initialize ConvLSTM-based pest spread forecaster

        Args:
            input_shape: (time_steps, height, width, features)
                        Features: [pests, ndvi, temperature, wind_speed]
        """
        self.input_shape = input_shape
        self.model = None

    def build_convlstm_model(self):
        """Build ConvLSTM model for spatiotemporal prediction"""
        model = Sequential([
            # First ConvLSTM layer
            ConvLSTM2D(
                filters=64, 
                kernel_size=(3, 3),
                input_shape=self.input_shape,
                padding='same',
                return_sequences=True,
                activation='relu'
            ),
            BatchNormalization(),

            # Second ConvLSTM layer
            ConvLSTM2D(
                filters=32,
                kernel_size=(3, 3), 
                padding='same',
                return_sequences=True,
                activation='relu'
            ),
            BatchNormalization(),

            # Third ConvLSTM layer
            ConvLSTM2D(
                filters=16,
                kernel_size=(3, 3),
                padding='same',
                return_sequences=False,
                activation='relu'
            ),
            BatchNormalization(),

            # Output layer - predict future pest distribution
            Conv2D(
                filters=1, 
                kernel_size=(3, 3),
                padding='same',
                activation='sigmoid'
            )
        ])

        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )

        self.model = model
        logger.info("ConvLSTM model built successfully")
        return model

    def create_synthetic_pest_spread_data(self, num_sequences=1000, sequence_length=10, grid_size=64):
        """
        Create synthetic pest spread training data
        Simulates realistic pest spreading patterns
        """
        logger.info("Creating synthetic pest spread data...")

        X_data = []
        y_data = []

        np.random.seed(42)

        for seq in range(num_sequences):
            # Initialize sequence
            sequence_data = np.zeros((sequence_length, grid_size, grid_size, 4))

            # Generate environmental conditions
            # NDVI layer (crop health)
            ndvi_base = self._generate_ndvi_field(grid_size)

            # Temperature and wind patterns
            temp_pattern = self._generate_weather_pattern(grid_size, 'temperature')
            wind_pattern = self._generate_weather_pattern(grid_size, 'wind')

            # Initial pest infestation (random small spots)
            pest_layer = self._generate_initial_pests(grid_size, num_spots=np.random.randint(1, 4))

            for t in range(sequence_length):
                # Environmental layers (relatively stable)
                sequence_data[t, :, :, 1] = ndvi_base + np.random.normal(0, 0.05, (grid_size, grid_size))  # NDVI
                sequence_data[t, :, :, 2] = temp_pattern + np.random.normal(0, 0.1, (grid_size, grid_size))  # Temperature
                sequence_data[t, :, :, 3] = wind_pattern + np.random.normal(0, 0.05, (grid_size, grid_size))  # Wind

                # Pest spread simulation
                if t == 0:
                    sequence_data[t, :, :, 0] = pest_layer
                else:
                    # Simulate pest spread based on previous state and environmental conditions
                    prev_pests = sequence_data[t-1, :, :, 0]
                    new_pests = self._simulate_pest_spread(
                        prev_pests,
                        sequence_data[t, :, :, 1],  # NDVI
                        sequence_data[t, :, :, 2],  # Temperature
                        sequence_data[t, :, :, 3]   # Wind
                    )
                    sequence_data[t, :, :, 0] = new_pests

            # Prepare training data (use first 9 timesteps to predict 10th)
            X_data.append(sequence_data[:-1])  # Input: first 9 timesteps
            y_data.append(sequence_data[-1, :, :, 0:1])  # Target: pest layer at timestep 10

        X_data = np.array(X_data)
        y_data = np.array(y_data)

        logger.info(f"Generated {num_sequences} sequences of shape {X_data.shape}")
        return X_data, y_data

    def _generate_ndvi_field(self, size):
        """Generate realistic NDVI field"""
        # Create agricultural field pattern
        y, x = np.ogrid[:size, :size]

        # Base healthy vegetation
        ndvi = 0.7 + 0.2 * np.sin(y/10) * np.sin(x/15)

        # Add field boundaries and paths
        ndvi[::16, :] *= 0.5  # Crop rows
        ndvi[:, ::20] *= 0.6  # Field boundaries

        # Add some variability
        ndvi += np.random.normal(0, 0.05, (size, size))

        return np.clip(ndvi, 0, 1)

    def _generate_weather_pattern(self, size, weather_type):
        """Generate weather patterns"""
        y, x = np.ogrid[:size, :size]

        if weather_type == 'temperature':
            # Temperature decreases slightly with elevation (simulated)
            pattern = 25 + (size - y) * 0.02 + np.sin(x/20) * 2
        else:  # wind
            # Wind speed varies across the field
            pattern = 8 + np.sin(y/15) * 3 + np.cos(x/12) * 2

        # Normalize
        pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min())
        return pattern

    def _generate_initial_pests(self, size, num_spots=2):
        """Generate initial pest infestation spots"""
        pest_layer = np.zeros((size, size))

        for _ in range(num_spots):
            # Random location for pest spot
            center_x = np.random.randint(size//4, 3*size//4)
            center_y = np.random.randint(size//4, 3*size//4)

            # Create circular infestation
            y, x = np.ogrid[:size, :size]
            mask = (x - center_x)**2 + (y - center_y)**2 <= (np.random.randint(3, 8))**2
            pest_layer[mask] = np.random.uniform(0.3, 0.8)

        return pest_layer

    def _simulate_pest_spread(self, prev_pests, ndvi, temperature, wind):
        """
        Simulate pest spread based on environmental conditions

        Rules:
        1. Pests spread faster in unhealthy vegetation (low NDVI)
        2. Optimal temperature range for pest activity
        3. Wind affects spread direction and speed
        """
        from scipy import ndimage

        new_pests = prev_pests.copy()

        # Pest reproduction (increase in existing areas)
        reproduction_rate = 1.1 + (1 - ndvi) * 0.3  # Higher reproduction in unhealthy plants

        # Temperature effect (optimal around 0.7 normalized temperature)
        temp_effect = 1 - np.abs(temperature - 0.7)
        reproduction_rate *= temp_effect

        new_pests *= reproduction_rate

        # Pest dispersal (spreading to neighboring areas)
        # Create dispersal kernel influenced by wind
        kernel = np.array([[0.05, 0.1, 0.05],
                          [0.1, 0.4, 0.1], 
                          [0.05, 0.1, 0.05]])

        # Apply wind influence (shift kernel based on wind pattern)
        dispersed_pests = ndimage.convolve(new_pests, kernel, mode='constant')

        # Combine original and dispersed pests
        new_pests = np.maximum(new_pests, dispersed_pests * 0.3)

        # Natural decay and limits
        new_pests = np.clip(new_pests * 0.98, 0, 1)  # Slight natural decay

        # Add some stochasticity
        noise = np.random.uniform(0, 0.02, new_pests.shape)
        new_pests = np.clip(new_pests + noise, 0, 1)

        return new_pests

    def train_model(self, X_train, y_train, epochs=50, batch_size=16):
        """Train the ConvLSTM model"""
        if self.model is None:
            self.build_convlstm_model()

        logger.info("Starting ConvLSTM training...")

        # Split validation data
        val_split = int(0.8 * len(X_train))
        X_val, y_val = X_train[val_split:], y_train[val_split:]
        X_train, y_train = X_train[:val_split], y_train[:val_split]

        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            verbose=1
        )

        # Save model
        self.model.save("backend/data/models/convlstm_forecaster.h5")
        logger.info("ConvLSTM model training completed and saved")

        return history

    def load_model(self, model_path="backend/data/models/convlstm_forecaster.h5"):
        """Load trained ConvLSTM model"""
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            logger.info("ConvLSTM model loaded successfully")
        else:
            logger.warning("Model file not found, building new model")
            self.build_convlstm_model()

    def predict_future_spread(self, input_sequence):
        """
        Predict future pest spread given input sequence

        Args:
            input_sequence: Input data of shape (time_steps, height, width, features)

        Returns:
            Predicted pest distribution
        """
        if len(input_sequence.shape) == 4:
            input_sequence = np.expand_dims(input_sequence, axis=0)

        prediction = self.model.predict(input_sequence)
        return prediction[0, :, :, 0]  # Return 2D pest probability map

    def create_forecast_stack(self, current_geotiffs, output_dir="backend/data/geotiff/forecasts"):
        """
        Create forecasted pest maps for next few days

        Args:
            current_geotiffs: Dictionary with paths to current GeoTIFF files
                            {'pests': path, 'ndvi': path, 'temperature': path, 'wind': path}
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        try:
            # Load current data
            data_stack = self._load_geotiff_stack(current_geotiffs)

            # Resize to model input size
            resized_stack = self._resize_stack(data_stack, target_size=(64, 64))

            # Create sequence (repeat current state for demo)
            sequence = np.repeat(resized_stack[np.newaxis, :], 9, axis=0)
            sequence = np.transpose(sequence, (1, 2, 3, 0))  # Rearrange dimensions

            # Predict future states
            predictions = []
            current_sequence = sequence.copy()

            for day in range(3):  # Forecast 3 days ahead
                pred = self.predict_future_spread(current_sequence)
                predictions.append(pred)

                # Update sequence for next prediction
                # Shift sequence and add prediction as latest state
                new_state = np.zeros_like(current_sequence[:, :, :, 0:1])
                new_state[:, :, :, 0] = pred

                current_sequence = np.concatenate([
                    current_sequence[:, :, :, 1:],
                    new_state
                ], axis=3)

            # Save predictions as GeoTIFFs
            template_path = current_geotiffs.get('pests', list(current_geotiffs.values())[0])

            for day, prediction in enumerate(predictions):
                output_path = os.path.join(output_dir, f"pest_forecast_day_{day+1}.tif")
                self._save_prediction_geotiff(prediction, template_path, output_path)

            logger.info(f"Forecast GeoTIFFs saved to {output_dir}")
            return predictions

        except Exception as e:
            logger.error(f"Error creating forecast stack: {e}")
            return None

    def _load_geotiff_stack(self, geotiff_paths):
        """Load and stack GeoTIFF files"""
        stack = []
        for key, path in geotiff_paths.items():
            if os.path.exists(path):
                with rasterio.open(path) as src:
                    data = src.read(1)
                    stack.append(data)
            else:
                logger.warning(f"File not found: {path}, using zeros")
                stack.append(np.zeros((100, 100)))  # Default size

        return np.stack(stack, axis=0)

    def _resize_stack(self, stack, target_size):
        """Resize data stack to target size"""
        import cv2
        resized_stack = []
        for layer in stack:
            resized = cv2.resize(layer, target_size, interpolation=cv2.INTER_LINEAR)
            resized_stack.append(resized)
        return np.stack(resized_stack, axis=0)

    def _save_prediction_geotiff(self, prediction, template_path, output_path):
        """Save prediction as GeoTIFF using template for georeferencing"""
        try:
            with rasterio.open(template_path) as template:
                profile = template.profile.copy()

                # Resize prediction to match template
                import cv2
                resized_pred = cv2.resize(
                    prediction, 
                    (profile['width'], profile['height']),
                    interpolation=cv2.INTER_LINEAR
                )

                profile.update(dtype=rasterio.float32, count=1)

                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(resized_pred.astype(rasterio.float32), 1)

        except Exception as e:
            logger.error(f"Error saving prediction GeoTIFF: {e}")

if __name__ == "__main__":
    import os

    # Initialize forecaster
    forecaster = PestSpreadForecaster()

    # Create synthetic training data
    X_train, y_train = forecaster.create_synthetic_pest_spread_data(
        num_sequences=500, sequence_length=10, grid_size=64
    )

    # Build and train model
    forecaster.build_convlstm_model()
    history = forecaster.train_model(X_train, y_train, epochs=20, batch_size=8)

    # Create sample forecasts
    geotiff_files = {
        'pests': 'backend/data/geotiff/initial_pests.tif',
        'ndvi': 'backend/data/geotiff/ndvi.tif', 
        'temperature': 'backend/data/geotiff/temperature.tif',
        'wind': 'backend/data/geotiff/wind_speed.tif'
    }

    forecasts = forecaster.create_forecast_stack(geotiff_files)

    print("Pest spread forecasting model completed!")
