
"""
Phase 2: Plant Disease Detection using ResNet CNN
Train a ResNet model on Kaggle Plant Disease dataset to detect pest damage
"""
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import os
from PIL import Image
import rasterio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlantDiseaseDetector:
    def __init__(self, input_shape=(224, 224, 3), num_classes=2):
        """
        Initialize Plant Disease Detector using ResNet

        Args:
            input_shape: Input image shape
            num_classes: Number of classes (healthy vs diseased)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None

    def build_resnet_model(self):
        """Build ResNet-50 based model for plant disease detection"""
        # Load pre-trained ResNet50 without top layers
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )

        # Freeze base model layers (transfer learning)
        base_model.trainable = False

        # Add custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)

        # Output layer
        predictions = Dense(self.num_classes, activation='softmax', name='predictions')(x)

        # Create the model
        self.model = Model(inputs=base_model.input, outputs=predictions)

        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )

        logger.info("ResNet model built successfully")
        return self.model

    def create_synthetic_training_data(self, num_samples=1000):
        """
        Create synthetic plant disease training data
        (In production, this would load the actual Kaggle Plant Disease dataset)
        """
        logger.info("Creating synthetic training data for demonstration...")

        # Create directories
        os.makedirs("backend/data/synthetic_dataset/train/healthy", exist_ok=True)
        os.makedirs("backend/data/synthetic_dataset/train/diseased", exist_ok=True)
        os.makedirs("backend/data/synthetic_dataset/validation/healthy", exist_ok=True)
        os.makedirs("backend/data/synthetic_dataset/validation/diseased", exist_ok=True)

        np.random.seed(42)

        for split in ['train', 'validation']:
            samples_per_split = num_samples if split == 'train' else num_samples // 4

            for class_name in ['healthy', 'diseased']:
                for i in range(samples_per_split // 2):
                    # Create synthetic plant leaf images
                    img = self._generate_synthetic_leaf_image(class_name)

                    # Save image
                    img_path = f"backend/data/synthetic_dataset/{split}/{class_name}/img_{i:04d}.jpg"
                    cv2.imwrite(img_path, img)

        logger.info("Synthetic training data created")

    def _generate_synthetic_leaf_image(self, class_type):
        """Generate synthetic leaf image"""
        # Create base leaf-like shape
        img = np.zeros((224, 224, 3), dtype=np.uint8)

        if class_type == 'healthy':
            # Healthy leaf - green with natural variations
            base_color = [34, 139, 34]  # Forest green
            noise = np.random.normal(0, 15, (224, 224, 3))

        else:  # diseased
            # Diseased leaf - brown/yellow spots, different coloration
            base_color = [85, 107, 47]  # Dark olive green
            noise = np.random.normal(0, 20, (224, 224, 3))

            # Add disease spots
            for _ in range(np.random.randint(3, 8)):
                center = (np.random.randint(50, 174), np.random.randint(50, 174))
                radius = np.random.randint(10, 30)
                color = [139, 69, 19] if np.random.random() > 0.5 else [255, 255, 0]  # Brown or yellow
                cv2.circle(img, center, radius, color, -1)

        # Apply base color and noise
        img = np.clip(np.array(base_color) + noise, 0, 255).astype(np.uint8)

        # Add leaf-like shape using ellipse
        center = (112, 112)
        axes = (80, 60)
        angle = np.random.randint(-45, 45)

        # Create mask for leaf shape
        mask = np.zeros((224, 224), dtype=np.uint8)
        cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)

        # Apply mask
        img = cv2.bitwise_and(img, img, mask=mask)

        return img

    def train_model(self, epochs=10):
        """Train the ResNet model"""
        if self.model is None:
            self.build_resnet_model()

        # Create data generators
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        validation_datagen = ImageDataGenerator(rescale=1./255)

        # Load training data
        train_generator = train_datagen.flow_from_directory(
            'backend/data/synthetic_dataset/train',
            target_size=self.input_shape[:2],
            batch_size=32,
            class_mode='categorical'
        )

        validation_generator = validation_datagen.flow_from_directory(
            'backend/data/synthetic_dataset/validation',
            target_size=self.input_shape[:2],
            batch_size=32,
            class_mode='categorical'
        )

        # Train model
        logger.info("Starting model training...")

        history = self.model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // train_generator.batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // validation_generator.batch_size,
            verbose=1
        )

        # Save model
        self.model.save("backend/data/models/resnet_plant_disease_model.h5")
        logger.info("Model training completed and saved")

        return history

    def load_model(self, model_path="backend/data/models/resnet_plant_disease_model.h5"):
        """Load trained model"""
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            logger.info("Model loaded successfully")
        else:
            logger.warning("Model file not found, building new model")
            self.build_resnet_model()

    def predict_image_patches(self, image, patch_size=224, overlap=0.5):
        """
        Predict on image patches to create pest infestation map

        Args:
            image: Input image (numpy array or path)
            patch_size: Size of patches for classification
            overlap: Overlap between patches (0-1)

        Returns:
            Prediction map showing pest probability for each patch
        """
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        height, width = image.shape[:2]
        step_size = int(patch_size * (1 - overlap))

        predictions = []
        coordinates = []

        for y in range(0, height - patch_size + 1, step_size):
            for x in range(0, width - patch_size + 1, step_size):
                # Extract patch
                patch = image[y:y+patch_size, x:x+patch_size]

                # Preprocess patch
                patch = cv2.resize(patch, self.input_shape[:2])
                patch = patch.astype(np.float32) / 255.0
                patch = np.expand_dims(patch, axis=0)

                # Predict
                pred = self.model.predict(patch, verbose=0)
                pest_probability = pred[0][1]  # Probability of diseased class

                predictions.append(pest_probability)
                coordinates.append((x, y))

        return np.array(predictions), coordinates

    def create_pest_map_geotiff(self, drone_image_path, output_path, bbox=None):
        """
        Create pest map GeoTIFF from drone image

        Args:
            drone_image_path: Path to drone image
            output_path: Output path for pest map GeoTIFF
            bbox: Geographic bounding box [min_lon, min_lat, max_lon, max_lat]
        """
        try:
            # Load drone image
            drone_image = cv2.imread(drone_image_path)
            if drone_image is None:
                # Create sample drone image for demo
                drone_image = self._create_sample_drone_image()
            else:
                drone_image = cv2.cvtColor(drone_image, cv2.COLOR_BGR2RGB)

            # Predict on image patches
            predictions, coordinates = self.predict_image_patches(drone_image)

            # Create prediction grid
            height, width = drone_image.shape[:2]
            prediction_grid = np.zeros((height//224 + 1, width//224 + 1))

            for i, (x, y) in enumerate(coordinates):
                grid_x, grid_y = x//224, y//224
                prediction_grid[grid_y, grid_x] = predictions[i]

            # Resize prediction grid to match drone image resolution
            prediction_map = cv2.resize(
                prediction_grid, 
                (width, height), 
                interpolation=cv2.INTER_LINEAR
            )

            # Default bounding box if not provided
            if bbox is None:
                bbox = [-74.2, 40.7, -74.0, 40.9]  # Example coordinates

            # Create geospatial transform
            transform = rasterio.transform.from_bounds(
                bbox[0], bbox[1], bbox[2], bbox[3], width, height
            )

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
                dst.write(prediction_map.astype(rasterio.float32), 1)

            logger.info(f"Pest map GeoTIFF created: {output_path}")
            return prediction_map

        except Exception as e:
            logger.error(f"Error creating pest map: {e}")
            return None

    def _create_sample_drone_image(self, size=(1000, 1000)):
        """Create sample drone image for demonstration"""
        logger.info("Creating sample drone image...")

        # Create realistic agricultural field image
        img = np.zeros((*size, 3), dtype=np.uint8)

        # Base green field color
        img[:, :] = [34, 139, 34]  # Forest green

        # Add field patterns (crop rows)
        for i in range(0, size[0], 20):
            img[i:i+2, :] = [85, 107, 47]  # Darker green for crop rows

        # Add some diseased areas
        for _ in range(10):
            x, y = np.random.randint(100, size[0]-100), np.random.randint(100, size[1]-100)
            radius = np.random.randint(30, 80)
            cv2.circle(img, (y, x), radius, [139, 69, 19], -1)  # Brown diseased areas

        # Add noise for realism
        noise = np.random.normal(0, 10, img.shape)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)

        # Save sample image
        cv2.imwrite("backend/data/sample_drone_image.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        return img

if __name__ == "__main__":
    # Initialize detector
    detector = PlantDiseaseDetector()

    # Create synthetic training data (replace with Kaggle dataset loading)
    detector.create_synthetic_training_data(num_samples=800)

    # Build and train model
    detector.build_resnet_model()
    history = detector.train_model(epochs=5)  # Reduced for demo

    # Create pest map from drone image
    pest_map = detector.create_pest_map_geotiff(
        "backend/data/sample_drone_image.jpg",
        "backend/data/geotiff/initial_pests.tif"
    )

    print("Plant disease detection model training completed!")
