from datetime import datetime
from enum import Enum, auto
from typing import Optional, List, Dict, Any, Union


class UserStatus(str, Enum):
    """User account status"""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DELETED = "deleted"


class PredictionStatus(str, Enum):
    """Prediction status"""
    PENDING = "pending"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class BetStatus(str, Enum):
    """Bet status"""
    PLACED = "placed"
    SETTLED = "settled"
    CANCELLED = "cancelled"


class BetOutcome(str, Enum):
    """Bet outcome"""
    WON = "won"
    LOST = "lost"


class TransactionType(str, Enum):
    """Transaction type"""
    BET = "bet"
    REWARD = "reward"
    REFUND = "refund"
    MINT = "mint"
    BURN = "burn"
    TRANSFER = "transfer"


class User:
    """User model"""

    def __init__(
            self,
            user_id: str,
            username: str,
            wallet_address: str,
            email: Optional[str] = None,
            created_at: Union[str, datetime] = None,
            last_login: Optional[Union[str, datetime]] = None,
            status: str = UserStatus.ACTIVE,
            reputation_nfts: Optional[List[Dict[str, Any]]] = None
    ):
        self.user_id = user_id
        self.username = username
        self.wallet_address = wallet_address
        self.email = email

        if created_at is None:
            self.created_at = datetime.now().isoformat()
        elif isinstance(created_at, datetime):
            self.created_at = created_at.isoformat()
        else:
            self.created_at = created_at

        if isinstance(last_login, datetime):
            self.last_login = last_login.isoformat()
        else:
            self.last_login = last_login

        self.status = status
        self.reputation_nfts = reputation_nfts or []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "wallet_address": self.wallet_address,
            "email": self.email,
            "created_at": self.created_at,
            "last_login": self.last_login,
            "status": self.status,
            "reputation_nfts": self.reputation_nfts
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'User':
        """Create from dictionary"""
        return cls(
            user_id=data["user_id"],
            username=data["username"],
            wallet_address=data["wallet_address"],
            email=data.get("email"),
            created_at=data["created_at"],
            last_login=data.get("last_login"),
            status=data.get("status", UserStatus.ACTIVE),
            reputation_nfts=data.get("reputation_nfts", [])
        )


class Prediction:
    """Prediction model"""

    def __init__(
            self,
            prediction_id: str,
            asset: str,
            prediction_type: str,
            predicted_value: float,
            confidence: float,
            timestamp: Union[str, datetime],
            time_horizon: str,
            metadata: Optional[Dict[str, Any]] = None,
            actual_value: Optional[float] = None,
            outcome_timestamp: Optional[Union[str, datetime]] = None,
            status: str = PredictionStatus.PENDING
    ):
        self.prediction_id = prediction_id
        self.asset = asset
        self.prediction_type = prediction_type
        self.predicted_value = predicted_value
        self.confidence = confidence

        if isinstance(timestamp, datetime):
            self.timestamp = timestamp.isoformat()
        else:
            self.timestamp = timestamp

        self.time_horizon = time_horizon
        self.metadata = metadata or {}
        self.actual_value = actual_value

        if isinstance(outcome_timestamp, datetime):
            self.outcome_timestamp = outcome_timestamp.isoformat()
        else:
            self.outcome_timestamp = outcome_timestamp

        self.status = status

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "prediction_id": self.prediction_id,
            "asset": self.asset,
            "prediction_type": self.prediction_type,
            "predicted_value": self.predicted_value,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "time_horizon": self.time_horizon,
            "metadata": self.metadata,
            "actual_value": self.actual_value,
            "outcome_timestamp": self.outcome_timestamp,
            "status": self.status
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Prediction':
        """Create from dictionary"""
        return cls(
            prediction_id=data["prediction_id"],
            asset=data["asset"],
            prediction_type=data["prediction_type"],
            predicted_value=data["predicted_value"],
            confidence=data["confidence"],
            timestamp=data["timestamp"],
            time_horizon=data["time_horizon"],
            metadata=data.get("metadata", {}),
            actual_value=data.get("actual_value"),
            outcome_timestamp=data.get("outcome_timestamp"),
            status=data.get("status", PredictionStatus.PENDING)
        )


class Bet:
    """Bet model"""

    def __init__(
            self,
            bet_id: str,
            user_id: str,
            prediction_id: str,
            amount: float,
            timestamp: Union[str, datetime],
            expected_value: Optional[float] = None,
            direction: Optional[str] = None,
            status: str = BetStatus.PLACED,
            outcome: Optional[str] = None,
            reward_amount: Optional[float] = None,
            transaction_signature: Optional[str] = None,
            reward_transaction_signature: Optional[str] = None,
            refund_transaction_signature: Optional[str] = None,
            settled_at: Optional[Union[str, datetime]] = None,
            cancelled_at: Optional[Union[str, datetime]] = None
    ):
        self.bet_id = bet_id
        self.user_id = user_id
        self.prediction_id = prediction_id
        self.amount = amount

        if isinstance(timestamp, datetime):
            self.timestamp = timestamp.isoformat()
        else:
            self.timestamp = timestamp

        self.expected_value = expected_value
        self.direction = direction
        self.status = status
        self.outcome = outcome
        self.reward_amount = reward_amount
        self.transaction_signature = transaction_signature
        self.reward_transaction_signature = reward_transaction_signature
        self.refund_transaction_signature = refund_transaction_signature

        if isinstance(settled_at, datetime):
            self.settled_at = settled_at.isoformat()
        else:
            self.settled_at = settled_at

        if isinstance(cancelled_at, datetime):
            self.cancelled_at = cancelled_at.isoformat()
        else:
            self.cancelled_at = cancelled_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "bet_id": self.bet_id,
            "user_id": self.user_id,
            "prediction_id": self.prediction_id,
            "amount": self.amount,
            "timestamp": self.timestamp,
            "expected_value": self.expected_value,
            "direction": self.direction,
            "status": self.status,
            "outcome": self.outcome,
            "reward_amount": self.reward_amount,
            "transaction_signature": self.transaction_signature,
            "reward_transaction_signature": self.reward_transaction_signature,
            "refund_transaction_signature": self.refund_transaction_signature,
            "settled_at": self.settled_at,
            "cancelled_at": self.cancelled_at
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Bet':
        """Create from dictionary"""
        return cls(
            bet_id=data["bet_id"],
            user_id=data["user_id"],
            prediction_id=data["prediction_id"],
            amount=data["amount"],
            timestamp=data["timestamp"],
            expected_value=data.get("expected_value"),
            direction=data.get("direction"),
            status=data.get("status", BetStatus.PLACED),
            outcome=data.get("outcome"),
            reward_amount=data.get("reward_amount"),
            transaction_signature=data.get("transaction_signature"),
            reward_transaction_signature=data.get("reward_transaction_signature"),
            refund_transaction_signature=data.get("refund_transaction_signature"),
            settled_at=data.get("settled_at"),
            cancelled_at=data.get("cancelled_at")
        )


class Transaction:
    """Transaction model"""

    def __init__(
            self,
            signature: str,
            user: str,
            amount: float,
            prediction_id: Optional[str] = None,
            timestamp: Union[str, datetime] = None,
            metadata: Optional[Dict[str, Any]] = None,
            status: str = "confirmed",
            type: str = TransactionType.BET
    ):
        self.signature = signature
        self.user = user
        self.amount = amount
        self.prediction_id = prediction_id

        if timestamp is None:
            self.timestamp = datetime.now().isoformat()
        elif isinstance(timestamp, datetime):
            self.timestamp = timestamp.isoformat()
        else:
            self.timestamp = timestamp

        self.metadata = metadata or {}
        self.status = status
        self.type = type

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "signature": self.signature,
            "user": self.user,
            "amount": self.amount,
            "prediction_id": self.prediction_id,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "status": self.status,
            "type": self.type
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Transaction':
        """Create from dictionary"""
        return cls(
            signature=data["signature"],
            user=data["user"],
            amount=data["amount"],
            prediction_id=data.get("prediction_id"),
            timestamp=data["timestamp"],
            metadata=data.get("metadata", {}),
            status=data.get("status", "confirmed"),
            type=data.get("type", TransactionType.BET)
        )