import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ai.models import ModelFactory, PredictionType
from ai.data_processor import DataProcessor
from utils.logging import get_logger
from config import CONFIG

logger = get_logger(__name__)


class ModelTrainer:
    """Responsible for training and updating prediction models"""

    def __init__(self, prediction_type, data_processor=None):
        self.prediction_type = prediction_type
        self.data_processor = data_processor or DataProcessor()

    def train_model(self, asset, time_horizon="24h", **kwargs):
        """
        Train a model for the specified asset and time horizon

        Args:
            asset: Asset to train model for (crypto symbol, NFT collection, etc.)
            time_horizon: Time frame for prediction ("24h", "7d", etc.)
            **kwargs: Additional parameters for training

        Returns:
            bool: Success status
        """
        try:
            # Get appropriate model
            model = ModelFactory.get_model(
                self.prediction_type,
                asset=asset,
                time_horizon=time_horizon,
                version=kwargs.get("version", "1.0.0")
            )

            # Fetch and prepare training data
            X, y = self._prepare_training_data(asset, time_horizon, **kwargs)

            # Split into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Train the model
            model.train(X_train, y_train)

            # Evaluate the model
            score = self._evaluate_model(model, X_val, y_val)

            logger.info(f"Model trained successfully with validation score: {score}")
            return True

        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return False

    def _prepare_training_data(self, asset, time_horizon, **kwargs):
        """
        Prepare data for training the model

        Args:
            asset: Asset identifier
            time_horizon: Time frame for prediction
            **kwargs: Additional parameters

        Returns:
            tuple: (X, y) feature matrix and target vector
        """
        # Check if using historic or custom data
        if "data" in kwargs and isinstance(kwargs["data"], pd.DataFrame):
            # Use provided data
            data = kwargs["data"]
        else:
            # Fetch historical data
            end_date = kwargs.get("end_date", datetime.now())
            days = int(kwargs.get("days", 365))
            start_date = end_date - timedelta(days=days)

            data = self.data_processor.fetch_historical_data(
                asset, start_date, end_date
            )

        # Process data based on prediction type
        if self.prediction_type == PredictionType.PRICE_DIRECTION:
            X, y = self._prepare_price_direction_data(data, time_horizon)
        elif self.prediction_type == PredictionType.PRICE_TARGET:
            X, y = self._prepare_price_target_data(data, time_horizon)
        elif self.prediction_type == PredictionType.NFT_FLOOR_PRICE:
            X, y = self._prepare_nft_floor_price_data(data, time_horizon)
        else:
            raise ValueError(f"Unsupported prediction type: {self.prediction_type}")

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        return X_scaled, y

    def _prepare_price_direction_data(self, data, time_horizon):
        """Prepare data for price direction prediction"""
        # Calculate price direction (1 for up, 0 for down)
        # Based on time_horizon
        periods = self._time_horizon_to_periods(time_horizon, data)

        # Create target: price direction after specified periods
        data['future_price'] = data['close'].shift(-periods)
        data['direction'] = (data['future_price'] > data['close']).astype(int)

        # Create features
        features = self.data_processor.create_price_features(data)

        # Drop rows with NaN values (due to shifting)
        features = features.dropna()
        y = features['direction']
        X = features.drop(['direction', 'future_price'], axis=1, errors='ignore')

        return X, y

    def _prepare_price_target_data(self, data, time_horizon):
        """Prepare data for price target prediction"""
        # Based on time_horizon
        periods = self._time_horizon_to_periods(time_horizon, data)

        # Create target: future price
        data['future_price'] = data['close'].shift(-periods)

        # Create features
        features = self.data_processor.create_price_features(data)

        # Drop rows with NaN values
        features = features.dropna()
        y = features['future_price']
        X = features.drop(['future_price'], axis=1, errors='ignore')

        return X, y

    def _prepare_nft_floor_price_data(self, data, time_horizon):
        """Prepare data for NFT floor price prediction"""
        # Similar to price target prediction but for NFT floor prices
        periods = self._time_horizon_to_periods(time_horizon, data)

        # Create target: future floor price
        data['future_floor_price'] = data['floor_price'].shift(-periods)

        # Create features
        features = self.data_processor.create_nft_features(data)

        # Drop rows with NaN values
        features = features.dropna()
        y = features['future_floor_price']
        X = features.drop(['future_floor_price'], axis=1, errors='ignore')

        return X, y

    def _time_horizon_to_periods(self, time_horizon, data):
        """Convert time horizon (e.g., '24h', '7d') to number of periods in the data"""
        # Determine period frequency of the data
        if 'hour' in data.index.name or data.index.freq == 'H':
            # Hourly data
            if time_horizon.endswith('h'):
                return int(time_horizon[:-1])
            elif time_horizon.endswith('d'):
                return int(time_horizon[:-1]) * 24
            else:
                return 24  # Default to 24 hours
        else:
            # Daily data
            if time_horizon.endswith('d'):
                return int(time_horizon[:-1])
            elif time_horizon.endswith('h'):
                return max(1, int(time_horizon[:-1]) // 24)
            else:
                return 1  # Default to 1 day

    def _evaluate_model(self, model, X_val, y_val):
        """Evaluate the trained model on validation data"""
        if self.prediction_type == PredictionType.PRICE_DIRECTION:
            # For classification, return accuracy
            predictions = model.predict(X_val)
            predictions_binary = np.array([1 if p >= 0.5 else 0 for p in predictions])
            accuracy = np.mean(predictions_binary == y_val)
            return accuracy
        else:
            # For regression, return mean absolute percentage error
            predictions = model.predict(X_val)
            mape = np.mean(np.abs((y_val - predictions) / y_val)) * 100
            return 100 - mape  # Return as a "score" (higher is better)

    def schedule_training(self, assets, time_horizons=None):
        """
        Schedule regular training for multiple assets and time horizons

        Args:
            assets: List of assets to train models for
            time_horizons: List of time horizons to train for (default: ["24h", "7d"])

        Returns:
            dict: Schedule information
        """
        if time_horizons is None:
            time_horizons = ["24h", "7d"]

        schedule_info = {
            "prediction_type": self.prediction_type.value,
            "assets": assets,
            "time_horizons": time_horizons,
            "interval": CONFIG.MODEL_TRAINING_INTERVAL,
            "next_training": datetime.now() + timedelta(seconds=CONFIG.MODEL_TRAINING_INTERVAL)
        }

        logger.info(f"Scheduled training for {len(assets)} assets with {len(time_horizons)} time horizons")
        return schedule_info


class TrainingManager:
    """Manages training of all prediction models"""

    def __init__(self, data_processor=None):
        self.data_processor = data_processor or DataProcessor()
        self.trainers = {
            PredictionType.PRICE_DIRECTION: ModelTrainer(PredictionType.PRICE_DIRECTION, self.data_processor),
            PredictionType.PRICE_TARGET: ModelTrainer(PredictionType.PRICE_TARGET, self.data_processor),
            PredictionType.NFT_FLOOR_PRICE: ModelTrainer(PredictionType.NFT_FLOOR_PRICE, self.data_processor)
        }
        self.schedules = []

    def train_all_models(self, assets, time_horizons=None):
        """Train all model types for the given assets and time horizons"""
        if time_horizons is None:
            time_horizons = ["24h", "7d"]

        results = {}

        for prediction_type, trainer in self.trainers.items():
            results[prediction_type.value] = {}

            for asset in assets:
                asset_results = []

                for time_horizon in time_horizons:
                    success = trainer.train_model(asset, time_horizon)
                    asset_results.append({
                        "time_horizon": time_horizon,
                        "success": success
                    })

                results[prediction_type.value][asset] = asset_results

        return results

    def schedule_all_trainings(self, assets, time_horizons=None):
        """Schedule training for all model types"""
        self.schedules = []

        for prediction_type, trainer in self.trainers.items():
            schedule = trainer.schedule_training(assets, time_horizons)
            self.schedules.append(schedule)

        return self.schedules