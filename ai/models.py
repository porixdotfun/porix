from enum import Enum
from abc import ABC
import os
import pickle
import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from config import CONFIG
from utils.logging import get_logger

logger = get_logger(__name__)


class PredictionType(Enum):
    PRICE_DIRECTION = "price_direction"  # Up or down
    PRICE_TARGET = "price_target"  # Specific price target
    VOLUME_CHANGE = "volume_change"  # Trading volume change
    NFT_FLOOR_PRICE = "nft_floor_price"  # NFT floor price
    EVENT_OCCURRENCE = "event_occurrence"  # Whether an event will occur


class BaseModel(ABC):
    """Base class for all prediction models in Porix platform"""

    def __init__(self, name, prediction_type, version="1.0.0"):
        self.name = name
        self.prediction_type = prediction_type
        self.version = version
        self.model = None
        self.model_path = os.path.join(CONFIG.MODEL_DIR, f"{name}_{version}.model")

    def save(self):
        """Save model to disk"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump(self.model, f)
        logger.info(f"Model {self.name} v{self.version} saved to {self.model_path}")

    def load(self):
        """Load model from disk"""
        if os.path.exists(self.model_path):
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            logger.info(f"Model {self.name} v{self.version} loaded from {self.model_path}")
            return True
        else:
            logger.warning(f"Model file {self.model_path} not found")
            return False

    def predict(self, features):
        """Make predictions using the model"""
        if self.model is None:
            success = self.load()
            if not success:
                raise ValueError(f"Model {self.name} is not trained yet")

        # Perform prediction
        try:
            return self._predict_implementation(features)
        except Exception as e:
            logger.error(f"Error during prediction with model {self.name}: {str(e)}")
            raise

    def _predict_implementation(self, features):
        """Implementation of prediction logic - to be overridden by subclasses"""
        raise NotImplementedError


class PriceDirectionModel(BaseModel):
    """Model to predict price direction (up/down) for cryptocurrencies"""

    def __init__(self, asset="general", version="1.0.0"):
        super().__init__(f"price_direction_{asset}", PredictionType.PRICE_DIRECTION, version)
        self.asset = asset

    def train(self, X_train, y_train):
        """Train the model with historical data"""
        logger.info(f"Training price direction model for {self.asset}")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.model.fit(X_train, y_train)
        logger.info(f"Model training complete with {len(X_train)} samples")
        self.save()

    def _predict_implementation(self, features):
        """Predict price direction (1 for up, 0 for down)"""
        predictions = self.model.predict_proba(features)
        # Return probability of price going up (class 1)
        return predictions[:, 1]


class PriceTargetModel(BaseModel):
    """Model to predict specific price targets for cryptocurrencies"""

    def __init__(self, asset="general", time_horizon="24h", version="1.0.0"):
        super().__init__(
            f"price_target_{asset}_{time_horizon}",
            PredictionType.PRICE_TARGET,
            version
        )
        self.asset = asset
        self.time_horizon = time_horizon

    def train(self, X_train, y_train):
        """Train the price target prediction model"""
        logger.info(f"Training price target model for {self.asset} with {self.time_horizon} horizon")
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.model.fit(X_train, y_train)
        logger.info(f"Model training complete with {len(X_train)} samples")
        self.save()

    def _predict_implementation(self, features):
        """Predict price target"""
        return self.model.predict(features)


class DeepLearningPriceModel(BaseModel):
    """Deep learning model for price prediction using LSTM architecture"""

    def __init__(self, asset="general", time_horizon="24h", version="1.0.0"):
        super().__init__(
            f"deep_price_{asset}_{time_horizon}",
            PredictionType.PRICE_TARGET,
            version
        )
        self.asset = asset
        self.time_horizon = time_horizon

    def build_model(self, input_shape):
        """Build the LSTM model architecture"""
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mean_squared_error'
        )
        return model

    def train(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.2):
        """Train the deep learning model"""
        logger.info(f"Training deep learning price model for {self.asset}")

        # Reshape input data for LSTM [samples, time steps, features]
        if len(X_train.shape) < 3:
            # Assume the data is of shape [samples, features]
            # Reshape to [samples, 1, features]
            X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])

        # Build and compile the model
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.model = self.build_model(input_shape)

        # Early stopping to prevent overfitting
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )

        # Train the model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
        )

        logger.info(f"Deep learning model training complete with {len(X_train)} samples")

        # Save model
        model_path = os.path.join(CONFIG.MODEL_DIR, f"{self.name}_{self.version}")
        self.model.save(model_path)
        logger.info(f"Deep learning model saved to {model_path}")

    def load(self):
        """Load the deep learning model from disk"""
        model_path = os.path.join(CONFIG.MODEL_DIR, f"{self.name}_{self.version}")
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            logger.info(f"Deep learning model {self.name} v{self.version} loaded")
            return True
        else:
            logger.warning(f"Model directory {model_path} not found")
            return False

    def _predict_implementation(self, features):
        """Predict using the deep learning model"""
        # Reshape input if needed
        if len(features.shape) < 3:
            features = features.reshape(features.shape[0], 1, features.shape[1])

        return self.model.predict(features)


class NFTFloorPriceModel(BaseModel):
    """Model to predict NFT floor price movements"""

    def __init__(self, collection="general", time_horizon="24h", version="1.0.0"):
        super().__init__(
            f"nft_floor_{collection}_{time_horizon}",
            PredictionType.NFT_FLOOR_PRICE,
            version
        )
        self.collection = collection
        self.time_horizon = time_horizon

    def train(self, X_train, y_train):
        """Train the NFT floor price prediction model"""
        logger.info(f"Training NFT floor price model for {self.collection}")
        self.model = GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=4,
            random_state=42
        )
        self.model.fit(X_train, y_train)
        logger.info(f"Model training complete with {len(X_train)} samples")
        self.save()

    def _predict_implementation(self, features):
        """Predict NFT floor price"""
        return self.model.predict(features)


class ModelFactory:
    """Factory class to create and manage prediction models"""

    @staticmethod
    def get_model(model_type, **kwargs):
        """Get appropriate model based on prediction type"""
        if model_type == PredictionType.PRICE_DIRECTION:
            return PriceDirectionModel(**kwargs)
        elif model_type == PredictionType.PRICE_TARGET:
            if kwargs.get("deep_learning", False):
                return DeepLearningPriceModel(**kwargs)
            return PriceTargetModel(**kwargs)
        elif model_type == PredictionType.NFT_FLOOR_PRICE:
            return NFTFloorPriceModel(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")