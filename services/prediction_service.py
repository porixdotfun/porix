import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import uuid
import json

from ai.predictors import PredictorFactory, PredictionType
from ai.data_processor import DataProcessor
from db.repositories import PredictionRepository
from utils.logging import get_logger
from config import CONFIG

logger = get_logger(__name__)


class PredictionService:
    """Service for managing predictions"""

    def __init__(self, prediction_repo=None, data_processor=None):
        self.prediction_repo = prediction_repo or PredictionRepository()
        self.data_processor = data_processor or DataProcessor()
        self.predictors = {
            PredictionType.PRICE_DIRECTION: PredictorFactory.get_predictor(PredictionType.PRICE_DIRECTION),
            PredictionType.PRICE_TARGET: PredictorFactory.get_predictor(PredictionType.PRICE_TARGET),
            PredictionType.NFT_FLOOR_PRICE: PredictorFactory.get_predictor(PredictionType.NFT_FLOOR_PRICE)
        }

    def predict_price_direction(self, asset, time_horizon="24h"):
        """
        Predict price direction for an asset

        Args:
            asset: Asset symbol
            time_horizon: Time frame for prediction ("24h", "7d", etc.)

        Returns:
            dict: Prediction result
        """
        logger.info(f"Predicting price direction for {asset} with horizon {time_horizon}")

        try:
            # Get feature data
            features = self._prepare_features(asset, time_horizon)

            # Get predictor
            predictor = self.predictors[PredictionType.PRICE_DIRECTION]

            # Make prediction
            result = predictor.predict(asset, features, time_horizon)

            # Store prediction in repository
            stored_prediction = self.prediction_repo.create_prediction(result.to_dict())

            # Format response
            response = {
                "prediction_id": result.prediction_id,
                "asset": asset,
                "prediction_type": "price_direction",
                "time_horizon": time_horizon,
                "predicted_direction": "up" if result.predicted_value >= 0.5 else "down",
                "probability_up": float(result.predicted_value),
                "confidence": float(result.confidence),
                "timestamp": result.timestamp.isoformat() if isinstance(result.timestamp,
                                                                        datetime) else result.timestamp,
                "expiry": (result.timestamp + self._parse_time_horizon(time_horizon)).isoformat() if isinstance(
                    result.timestamp, datetime) else None
            }

            return response

        except Exception as e:
            logger.error(f"Error predicting price direction: {str(e)}")
            return {
                "error": "Prediction failed",
                "message": str(e)
            }

    def predict_price_target(self, asset, time_horizon="24h"):
        """
        Predict price target for an asset

        Args:
            asset: Asset symbol
            time_horizon: Time frame for prediction ("24h", "7d", etc.)

        Returns:
            dict: Prediction result
        """
        logger.info(f"Predicting price target for {asset} with horizon {time_horizon}")

        try:
            # Get feature data
            features = self._prepare_features(asset, time_horizon)

            # Get predictor
            predictor = self.predictors[PredictionType.PRICE_TARGET]

            # Make prediction
            result = predictor.predict(asset, features, time_horizon)

            # Get current price
            current_price = self._get_current_price(asset)

            # Store prediction in repository
            stored_prediction = self.prediction_repo.create_prediction(result.to_dict())

            # Calculate percent change
            percent_change = (result.predicted_value - current_price) / current_price * 100

            # Format response
            response = {
                "prediction_id": result.prediction_id,
                "asset": asset,
                "prediction_type": "price_target",
                "time_horizon": time_horizon,
                "current_price": float(current_price),
                "predicted_price": float(result.predicted_value),
                "percent_change": float(percent_change),
                "confidence": float(result.confidence),
                "timestamp": result.timestamp.isoformat() if isinstance(result.timestamp,
                                                                        datetime) else result.timestamp,
                "expiry": (result.timestamp + self._parse_time_horizon(time_horizon)).isoformat() if isinstance(
                    result.timestamp, datetime) else None
            }

            return response

        except Exception as e:
            logger.error(f"Error predicting price target: {str(e)}")
            return {
                "error": "Prediction failed",
                "message": str(e)
            }

    def predict_nft_floor_price(self, collection, time_horizon="24h"):
        """
        Predict NFT floor price

        Args:
            collection: NFT collection name or ID
            time_horizon: Time frame for prediction ("24h", "7d", etc.)

        Returns:
            dict: Prediction result
        """
        logger.info(f"Predicting NFT floor price for {collection} with horizon {time_horizon}")

        try:
            # Add NFT: prefix for data processor
            asset = f"NFT:{collection}"

            # Get feature data
            features = self._prepare_features(asset, time_horizon)

            # Get predictor
            predictor = self.predictors[PredictionType.NFT_FLOOR_PRICE]

            # Make prediction
            result = predictor.predict(collection, features, time_horizon)

            # Get current floor price
            current_floor = self._get_current_floor_price(collection)

            # Store prediction in repository
            stored_prediction = self.prediction_repo.create_prediction(result.to_dict())

            # Calculate percent change
            percent_change = (result.predicted_value - current_floor) / current_floor * 100

            # Format response
            response = {
                "prediction_id": result.prediction_id,
                "collection": collection,
                "prediction_type": "nft_floor_price",
                "time_horizon": time_horizon,
                "current_floor_price": float(current_floor),
                "predicted_floor_price": float(result.predicted_value),
                "percent_change": float(percent_change),
                "confidence": float(result.confidence),
                "timestamp": result.timestamp.isoformat() if isinstance(result.timestamp,
                                                                        datetime) else result.timestamp,
                "expiry": (result.timestamp + self._parse_time_horizon(time_horizon)).isoformat() if isinstance(
                    result.timestamp, datetime) else None
            }

            return response

        except Exception as e:
            logger.error(f"Error predicting NFT floor price: {str(e)}")
            return {
                "error": "Prediction failed",
                "message": str(e)
            }

    def get_prediction(self, prediction_id):
        """
        Get a prediction by ID

        Args:
            prediction_id: Prediction ID

        Returns:
            dict: Prediction details
        """
        prediction = self.prediction_repo.get_prediction(prediction_id)

        if not prediction:
            return None

        return prediction

    def get_prediction_history(self, user_id=None, asset=None, prediction_type=None, limit=50):
        """
        Get prediction history

        Args:
            user_id: Filter by user ID
            asset: Filter by asset
            prediction_type: Filter by prediction type
            limit: Maximum number of results

        Returns:
            list: List of predictions
        """
        filters = {}

        if user_id:
            filters["user_id"] = user_id

        if asset:
            filters["asset"] = asset

        if prediction_type:
            filters["prediction_type"] = prediction_type

        predictions = self.prediction_repo.get_predictions(filters, limit)

        return predictions

    def get_prediction_outcome(self, prediction_id):
        """
        Get the outcome of a prediction

        Args:
            prediction_id: Prediction ID

        Returns:
            dict: Prediction outcome
        """
        # Get prediction
        prediction = self.prediction_repo.get_prediction(prediction_id)

        if not prediction:
            return None

        # Check if prediction has expired
        timestamp = datetime.fromisoformat(prediction["timestamp"]) if isinstance(prediction["timestamp"], str) else \
        prediction["timestamp"]
        time_horizon = prediction["time_horizon"]
        expiry = timestamp + self._parse_time_horizon(time_horizon)

        if datetime.now() < expiry:
            # Prediction hasn't expired yet
            return {
                "prediction_id": prediction_id,
                "status": "pending",
                "expiry": expiry.isoformat(),
                "time_remaining": (expiry - datetime.now()).total_seconds()
            }

        # Get actual outcome
        outcome = self._get_actual_outcome(prediction)

        # Update prediction with outcome
        self.prediction_repo.update_prediction(prediction_id, outcome)

        return {
            "prediction_id": prediction_id,
            "status": "completed",
            "expiry": expiry.isoformat(),
            "predicted_value": prediction["predicted_value"],
            "actual_value": outcome["actual_value"],
            "is_correct": outcome["is_correct"],
            "error_margin": outcome.get("error_margin")
        }

    def get_prediction_count(self):
        """
        Get total number of predictions

        Returns:
            int: Total prediction count
        """
        return self.prediction_repo.get_count()

    def _prepare_features(self, asset, time_horizon):
        """
        Prepare feature data for prediction

        Args:
            asset: Asset identifier
            time_horizon: Time frame for prediction

        Returns:
            numpy.ndarray: Feature matrix
        """
        # Get historical data
        end_date = datetime.now()
        # Use more historical data for longer horizons
        if time_horizon.endswith('d'):
            days = int(time_horizon[:-1]) * 7  # 7x the horizon in days
        else:
            days = int(time_horizon[:-1]) // 24 * 7 or 7  # 7x the horizon in days, min 7 days

        start_date = end_date - timedelta(days=days)

        # Fetch historical data
        data = self.data_processor.fetch_historical_data(asset, start_date, end_date)

        # Process features based on asset type
        if asset.startswith("NFT:"):
            # NFT collection
            features = self.data_processor.create_nft_features(data)
        else:
            # Cryptocurrency
            features = self.data_processor.create_price_features(data)

        # Drop non-feature columns
        drop_columns = ['direction', 'future_price', 'future_floor_price', 'open', 'high', 'low', 'close',
                        'floor_price']
        for col in drop_columns:
            if col in features.columns:
                features = features.drop(col, axis=1)

        # Handle NaN values
        features = features.dropna()

        # Convert to numpy array
        feature_matrix = features.values

        # Reshape for single prediction (add batch dimension)
        if len(feature_matrix.shape) == 1:
            feature_matrix = feature_matrix.reshape(1, -1)

        return feature_matrix

    def _get_current_price(self, asset):
        """
        Get current price for an asset

        Args:
            asset: Asset symbol

        Returns:
            float: Current price
        """
        # In a real implementation, this would call an API
        # Here we'll use our data processor to get the latest price

        end_date = datetime.now()
        start_date = end_date - timedelta(days=1)

        data = self.data_processor.fetch_historical_data(asset, start_date, end_date)

        if data.empty:
            # Default fallback price
            return 100.0 if asset == "PORIX" else 50000.0 if asset == "BTC" else 3000.0

        # Get the latest close price
        return data['close'].iloc[-1]

    def _get_current_floor_price(self, collection):
        """
        Get current floor price for an NFT collection

        Args:
            collection: NFT collection name or ID

        Returns:
            float: Current floor price
        """
        # In a real implementation, this would call an API
        # Here we'll use our data processor to get the latest floor price

        asset = f"NFT:{collection}"
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1)

        data = self.data_processor.fetch_historical_data(asset, start_date, end_date)

        if data.empty:
            # Default fallback price
            return 10.0 if "rare" in collection.lower() else 1.0

        # Get the latest floor price
        return data['floor_price'].iloc[-1]

    def _get_actual_outcome(self, prediction):
        """
        Get actual outcome for a prediction

        Args:
            prediction: Prediction data

        Returns:
            dict: Outcome data
        """
        prediction_type = prediction["prediction_type"]
        asset = prediction.get("asset", prediction.get("collection"))
        timestamp = datetime.fromisoformat(prediction["timestamp"]) if isinstance(prediction["timestamp"], str) else \
        prediction["timestamp"]
        time_horizon = prediction["time_horizon"]
        predicted_value = prediction["predicted_value"]

        # Calculate outcome timestamp
        outcome_timestamp = timestamp + self._parse_time_horizon(time_horizon)

        # Get actual value based on prediction type
        if prediction_type == "price_direction":
            # Get price at prediction time
            start_price = self._get_price_at_time(asset, timestamp)

            # Get price at outcome time
            end_price = self._get_price_at_time(asset, outcome_timestamp)

            # Determine actual direction (1 for up, 0 for down)
            actual_value = 1 if end_price > start_price else 0

            # Determine if prediction was correct
            predicted_direction = 1 if predicted_value >= 0.5 else 0
            is_correct = predicted_direction == actual_value

            return {
                "actual_value": actual_value,
                "is_correct": is_correct,
                "outcome_timestamp": outcome_timestamp.isoformat()
            }

        elif prediction_type == "price_target":
            # Get actual price at outcome time
            actual_value = self._get_price_at_time(asset, outcome_timestamp)

            # Calculate error margin
            error_margin = abs(predicted_value - actual_value) / actual_value * 100

            # Determine if prediction was within acceptable error margin (5%)
            is_correct = error_margin <= 5.0

            return {
                "actual_value": float(actual_value),
                "is_correct": is_correct,
                "error_margin": float(error_margin),
                "outcome_timestamp": outcome_timestamp.isoformat()
            }

        elif prediction_type == "nft_floor_price":
            # Get actual floor price at outcome time
            actual_value = self._get_floor_price_at_time(asset, outcome_timestamp)

            # Calculate error margin
            error_margin = abs(predicted_value - actual_value) / actual_value * 100

            # Determine if prediction was within acceptable error margin (10% for NFTs)
            is_correct = error_margin <= 10.0

            return {
                "actual_value": float(actual_value),
                "is_correct": is_correct,
                "error_margin": float(error_margin),
                "outcome_timestamp": outcome_timestamp.isoformat()
            }

        else:
            logger.error(f"Unknown prediction type: {prediction_type}")
            return {
                "actual_value": None,
                "is_correct": False,
                "outcome_timestamp": outcome_timestamp.isoformat()
            }

    def _get_price_at_time(self, asset, timestamp):
        """
        Get price of an asset at a specific time

        Args:
            asset: Asset symbol
            timestamp: Timestamp to get price for

        Returns:
            float: Price at the specified time
        """
        # In a real implementation, this would use historical data API
        # Here we'll simulate based on current price and random variations

        current_price = self._get_current_price(asset)
        time_diff = (datetime.now() - timestamp).total_seconds() / 86400  # days

        # Simulate price varying up to ±20% from current based on time difference
        np.random.seed(int(timestamp.timestamp()))
        variation = np.random.uniform(-0.2, 0.2) * min(time_diff, 10) / 10

        return current_price * (1 + variation)

    def _get_floor_price_at_time(self, collection, timestamp):
        """
        Get floor price of an NFT collection at a specific time

        Args:
            collection: NFT collection name or ID
            timestamp: Timestamp to get floor price for

        Returns:
            float: Floor price at the specified time
        """
        # In a real implementation, this would use historical data API
        # Here we'll simulate based on current floor price and random variations

        current_floor = self._get_current_floor_price(collection)
        time_diff = (datetime.now() - timestamp).total_seconds() / 86400  # days

        # Simulate floor price varying up to ±30% from current based on time difference
        np.random.seed(int(timestamp.timestamp()))
        variation = np.random.uniform(-0.3, 0.3) * min(time_diff, 10) / 10

        return current_floor * (1 + variation)

    def _parse_time_horizon(self, time_horizon):
        """
        Parse time horizon string to timedelta

        Args:
            time_horizon: Time horizon string (e.g., "24h", "7d")

        Returns:
            datetime.timedelta: Time delta
        """
        if time_horizon.endswith('h'):
            hours = int(time_horizon[:-1])
            return timedelta(hours=hours)
        elif time_horizon.endswith('d'):
            days = int(time_horizon[:-1])
            return timedelta(days=days)
        else:
            # Default to 24 hours
            logger.warning(f"Invalid time horizon format: {time_horizon}, using 24h")
            return timedelta(hours=24)