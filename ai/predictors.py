import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from abc import ABC

from ai.models import ModelFactory, PredictionType
from utils.logging import get_logger
from config import CONFIG

logger = get_logger(__name__)


class PredictionResult:
    """Represents the result of a prediction"""

    def __init__(self, prediction_id, asset, prediction_type,
                 predicted_value, confidence, timestamp, time_horizon,
                 metadata=None):
        self.prediction_id = prediction_id
        self.asset = asset
        self.prediction_type = prediction_type
        self.predicted_value = predicted_value
        self.confidence = confidence
        self.timestamp = timestamp
        self.time_horizon = time_horizon
        self.metadata = metadata or {}
        self.actual_value = None
        self.outcome_timestamp = None

    def to_dict(self):
        """Convert prediction result to dictionary"""
        return {
            "prediction_id": self.prediction_id,
            "asset": self.asset,
            "prediction_type": self.prediction_type.value if isinstance(self.prediction_type,
                                                                        PredictionType) else self.prediction_type,
            "predicted_value": float(self.predicted_value),
            "confidence": float(self.confidence),
            "timestamp": self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            "time_horizon": self.time_horizon,
            "metadata": self.metadata,
            "actual_value": float(self.actual_value) if self.actual_value is not None else None,
            "outcome_timestamp": self.outcome_timestamp.isoformat() if self.outcome_timestamp else None
        }

    def is_correct(self, threshold=0.05):
        """
        Determine if prediction was correct

        Args:
            threshold: Acceptable error threshold (percentage)

        Returns:
            bool: Whether prediction was correct within the threshold
        """
        if self.actual_value is None:
            return None

        if self.prediction_type == PredictionType.PRICE_DIRECTION:
            # For binary predictions (up/down), predicted_value is probability of going up
            # actual_value should be 1 (up) or 0 (down)
            predicted_direction = 1 if self.predicted_value >= 0.5 else 0
            return predicted_direction == self.actual_value
        else:
            # For numeric predictions, check if within threshold percentage
            error = abs(self.predicted_value - self.actual_value) / self.actual_value
            return error <= threshold


class BasePredictor(ABC):
    """Base class for all predictors in the Porix platform"""

    def __init__(self, prediction_type):
        self.prediction_type = prediction_type

    def predict(self, asset, features, time_horizon="24h", **kwargs):
        """
        Make a prediction for the given asset

        Args:
            asset: The asset to predict for (crypto symbol, nft collection, etc.)
            features: Feature vector or matrix for the prediction
            time_horizon: Time frame for the prediction (e.g., "24h", "7d")

        Returns:
            PredictionResult: The prediction result
        """
        prediction_id = self._generate_prediction_id(asset, time_horizon)
        model = self._get_model(asset, time_horizon)

        # Make prediction
        raw_prediction = model.predict(features)

        # Process prediction value
        predicted_value = self._process_prediction(raw_prediction)

        # Calculate confidence
        confidence = self._calculate_confidence(raw_prediction, features)

        # Create prediction result
        timestamp = datetime.now()
        result = PredictionResult(
            prediction_id=prediction_id,
            asset=asset,
            prediction_type=self.prediction_type,
            predicted_value=predicted_value,
            confidence=confidence,
            timestamp=timestamp,
            time_horizon=time_horizon,
            metadata=kwargs.get("metadata", {})
        )

        return result

    def _generate_prediction_id(self, asset, time_horizon):
        """Generate a unique ID for the prediction"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"{asset}-{self.prediction_type.value}-{time_horizon}-{timestamp}"

    def _get_model(self, asset, time_horizon):
        """Get the appropriate model for this prediction"""
        return ModelFactory.get_model(self.prediction_type, asset=asset, time_horizon=time_horizon)

    def _process_prediction(self, raw_prediction):
        """Process the raw prediction to get final value"""
        # Default implementation just takes the first value
        if isinstance(raw_prediction, np.ndarray):
            return raw_prediction[0]
        return raw_prediction

    def _calculate_confidence(self, raw_prediction, features):
        """Calculate confidence score for the prediction"""
        # Base implementation returns a fixed confidence
        return 0.8


class PriceDirectionPredictor(BasePredictor):
    """Predictor for price direction (up/down)"""

    def __init__(self):
        super().__init__(PredictionType.PRICE_DIRECTION)

    def _process_prediction(self, raw_prediction):
        """Process raw prediction (probability of going up)"""
        # Return probability of going up
        if isinstance(raw_prediction, np.ndarray) and len(raw_prediction) > 0:
            return float(raw_prediction[0])
        return float(raw_prediction)

    def _calculate_confidence(self, raw_prediction, features):
        """Calculate confidence based on how far from 0.5 the prediction is"""
        prob_up = self._process_prediction(raw_prediction)
        # Confidence is how far from uncertainty (0.5) the prediction is
        return 2 * abs(prob_up - 0.5)


class PriceTargetPredictor(BasePredictor):
    """Predictor for specific price targets"""

    def __init__(self, use_deep_learning=False):
        super().__init__(PredictionType.PRICE_TARGET)
        self.use_deep_learning = use_deep_learning

    def _get_model(self, asset, time_horizon):
        """Get the appropriate price target model"""
        return ModelFactory.get_model(
            self.prediction_type,
            asset=asset,
            time_horizon=time_horizon,
            deep_learning=self.use_deep_learning
        )

    def _calculate_confidence(self, raw_prediction, features):
        """Calculate confidence based on historical model performance"""
        # In real implementation, this would use historical model performance
        # Here using a placeholder confidence level
        return 0.75


class NFTFloorPricePredictor(BasePredictor):
    """Predictor for NFT floor prices"""

    def __init__(self):
        super().__init__(PredictionType.NFT_FLOOR_PRICE)

    def _get_model(self, collection, time_horizon):
        """Get the appropriate NFT floor price model"""
        return ModelFactory.get_model(
            self.prediction_type,
            collection=collection,
            time_horizon=time_horizon
        )

    def predict(self, collection, features, time_horizon="24h", **kwargs):
        """Override to use collection instead of asset"""
        return super().predict(collection, features, time_horizon, **kwargs)


class PredictorFactory:
    """Factory for creating predictors"""

    @staticmethod
    def get_predictor(prediction_type):
        """Get appropriate predictor based on prediction type"""
        if prediction_type == PredictionType.PRICE_DIRECTION:
            return PriceDirectionPredictor()
        elif prediction_type == PredictionType.PRICE_TARGET:
            return PriceTargetPredictor()
        elif prediction_type == PredictionType.NFT_FLOOR_PRICE:
            return NFTFloorPricePredictor()
        else:
            raise ValueError(f"Unknown prediction type: {prediction_type}")