from typing import Dict, List, Any, Optional
from datetime import datetime

from db.models import User, Prediction, Bet, Transaction
from db.connection import db
from utils.logging import get_logger

logger = get_logger(__name__)


class BaseRepository:
    """Base repository with common functionality"""

    def __init__(self, collection_name):
        self.collection_name = collection_name

    @property
    def collection(self):
        """Get the database collection"""
        return db[self.collection_name]

    def _apply_filters(self, filters):
        """
        Convert filter dict to MongoDB query

        Args:
            filters: Dictionary of filters

        Returns:
            dict: MongoDB query
        """
        query = {}

        # Special handling for timestamp filters
        if 'timestamp_after' in filters:
            timestamp = filters.pop('timestamp_after')
            query['timestamp'] = {'$gte': timestamp}

        if 'timestamp_before' in filters:
            timestamp = filters.pop('timestamp_before')
            if 'timestamp' in query:
                query['timestamp']['$lte'] = timestamp
            else:
                query['timestamp'] = {'$lte': timestamp}

        # Add remaining filters
        query.update(filters)

        return query


class UserRepository(BaseRepository):
    """Repository for user data"""

    def __init__(self):
        super().__init__('users')

    def create_user(self, user_data) -> Dict[str, Any]:
        """
        Create a new user

        Args:
            user_data: User data dictionary or User object

        Returns:
            dict: Created user
        """
        if isinstance(user_data, User):
            user_dict = user_data.to_dict()
        else:
            user_dict = user_data

        try:
            result = self.collection.insert_one(user_dict)

            if result.acknowledged:
                return user_dict
            else:
                logger.error("Failed to create user")
                return None

        except Exception as e:
            logger.error(f"Error creating user: {str(e)}")
            return None

    def get_user(self, user_id) -> Optional[Dict[str, Any]]:
        """
        Get a user by ID

        Args:
            user_id: User ID

        Returns:
            dict: User data or None if not found
        """
        user = self.collection.find_one({"user_id": user_id})
        return user

    def get_user_by_username(self, username) -> Optional[Dict[str, Any]]:
        """
        Get a user by username

        Args:
            username: Username

        Returns:
            dict: User data or None if not found
        """
        user = self.collection.find_one({"username": username})
        return user

    def get_user_by_wallet(self, wallet_address) -> Optional[Dict[str, Any]]:
        """
        Get a user by wallet address

        Args:
            wallet_address: Wallet address

        Returns:
            dict: User data or None if not found
        """
        user = self.collection.find_one({"wallet_address": wallet_address})
        return user

    def get_users(self, filters={}, limit=50) -> List[Dict[str, Any]]:
        """
        Get users with filters

        Args:
            filters: Dictionary of filters
            limit: Maximum number of results

        Returns:
            list: List of users
        """
        query = self._apply_filters(filters)
        users = list(self.collection.find(query).limit(limit))
        return users

    def update_user(self, user_id, updates) -> Dict[str, Any]:
        """
        Update a user

        Args:
            user_id: User ID
            updates: Fields to update

        Returns:
            dict: Updated user
        """
        result = self.collection.update_one(
            {"user_id": user_id},
            {"$set": updates}
        )

        if result.modified_count > 0:
            return self.get_user(user_id)
        else:
            logger.warning(f"No changes made to user {user_id}")
            return self.get_user(user_id)

    def delete_user(self, user_id) -> bool:
        """
        Delete a user

        Args:
            user_id: User ID

        Returns:
            bool: Success status
        """
        result = self.collection.delete_one({"user_id": user_id})
        return result.deleted_count > 0

    def get_count(self) -> int:
        """
        Get total number of users

        Returns:
            int: User count
        """
        return self.collection.count_documents({})


class PredictionRepository(BaseRepository):
    """Repository for prediction data"""

    def __init__(self):
        super().__init__('predictions')

    def create_prediction(self, prediction_data) -> Dict[str, Any]:
        """
        Create a new prediction

        Args:
            prediction_data: Prediction data dictionary or Prediction object

        Returns:
            dict: Created prediction
        """
        if isinstance(prediction_data, Prediction):
            prediction_dict = prediction_data.to_dict()
        else:
            prediction_dict = prediction_data

        try:
            result = self.collection.insert_one(prediction_dict)

            if result.acknowledged:
                return prediction_dict
            else:
                logger.error("Failed to create prediction")
                return None

        except Exception as e:
            logger.error(f"Error creating prediction: {str(e)}")
            return None

    def get_prediction(self, prediction_id) -> Optional[Dict[str, Any]]:
        """
        Get a prediction by ID

        Args:
            prediction_id: Prediction ID

        Returns:
            dict: Prediction data or None if not found
        """
        prediction = self.collection.find_one({"prediction_id": prediction_id})
        return prediction

    def get_predictions(self, filters={}, limit=50) -> List[Dict[str, Any]]:
        """
        Get predictions with filters

        Args:
            filters: Dictionary of filters
            limit: Maximum number of results

        Returns:
            list: List of predictions
        """
        query = self._apply_filters(filters)
        predictions = list(self.collection.find(query).limit(limit))
        return predictions

    def update_prediction(self, prediction_id, updates) -> Dict[str, Any]:
        """
        Update a prediction

        Args:
            prediction_id: Prediction ID
            updates: Fields to update

        Returns:
            dict: Updated prediction
        """
        result = self.collection.update_one(
            {"prediction_id": prediction_id},
            {"$set": updates}
        )

        if result.modified_count > 0:
            return self.get_prediction(prediction_id)
        else:
            logger.warning(f"No changes made to prediction {prediction_id}")
            return self.get_prediction(prediction_id)

    def get_count(self) -> int:
        """
        Get total number of predictions

        Returns:
            int: Prediction count
        """
        return self.collection.count_documents({})


class BetRepository(BaseRepository):
    """Repository for bet data"""

    def __init__(self):
        super().__init__('bets')

    def create_bet(self, bet_data) -> Dict[str, Any]:
        """
        Create a new bet

        Args:
            bet_data: Bet data dictionary or Bet object

        Returns:
            dict: Created bet
        """
        if isinstance(bet_data, Bet):
            bet_dict = bet_data.to_dict()
        else:
            bet_dict = bet_data

        try:
            result = self.collection.insert_one(bet_dict)

            if result.acknowledged:
                return bet_dict
            else:
                logger.error("Failed to create bet")
                return None

        except Exception as e:
            logger.error(f"Error creating bet: {str(e)}")
            return None

    def get_bet(self, bet_id) -> Optional[Dict[str, Any]]:
        """
        Get a bet by ID

        Args:
            bet_id: Bet ID

        Returns:
            dict: Bet data or None if not found
        """
        bet = self.collection.find_one({"bet_id": bet_id})
        return bet

    def get_bets(self, filters={}, limit=50) -> List[Dict[str, Any]]:
        """
        Get bets with filters

        Args:
            filters: Dictionary of filters
            limit: Maximum number of results

        Returns:
            list: List of bets
        """
        query = self._apply_filters(filters)
        bets = list(self.collection.find(query).limit(limit))
        return bets

    def get_bets_by_status(self, status) -> List[Dict[str, Any]]:
        """
        Get bets by status

        Args:
            status: Bet status

        Returns:
            list: List of bets
        """
        bets = list(self.collection.find({"status": status}))
        return bets

    def update_bet(self, bet_id, updates) -> Dict[str, Any]:
        """
        Update a bet

        Args:
            bet_id: Bet ID
            updates: Fields to update

        Returns:
            dict: Updated bet
        """
        result = self.collection.update_one(
            {"bet_id": bet_id},
            {"$set": updates}
        )

        if result.modified_count > 0:
            return self.get_bet(bet_id)
        else:
            logger.warning(f"No changes made to bet {bet_id}")
            return self.get_bet(bet_id)

    def get_count(self) -> int:
        """
        Get total number of bets

        Returns:
            int: Bet count
        """
        return self.collection.count_documents({})


class TransactionRepository(BaseRepository):
    """Repository for transaction data"""

    def __init__(self):
        super().__init__('transactions')

    def create_transaction(self, transaction_data) -> Dict[str, Any]:
        """
        Create a new transaction

        Args:
            transaction_data: Transaction data dictionary or Transaction object

        Returns:
            dict: Created transaction
        """
        if isinstance(transaction_data, Transaction):
            transaction_dict = transaction_data.to_dict()
        else:
            transaction_dict = transaction_data

        try:
            result = self.collection.insert_one(transaction_dict)

            if result.acknowledged:
                return transaction_dict
            else:
                logger.error("Failed to create transaction")
                return None

        except Exception as e:
            logger.error(f"Error creating transaction: {str(e)}")
            return None

    def get_transaction(self, signature) -> Optional[Dict[str, Any]]:
        """
        Get a transaction by signature

        Args:
            signature: Transaction signature

        Returns:
            dict: Transaction data or None if not found
        """
        transaction = self.collection.find_one({"signature": signature})
        return transaction

    def get_transactions(self, filters={}, limit=50) -> List[Dict[str, Any]]:
        """
        Get transactions with filters

        Args:
            filters: Dictionary of filters
            limit: Maximum number of results

        Returns:
            list: List of transactions
        """
        query = self._apply_filters(filters)
        transactions = list(self.collection.find(query).limit(limit))
        return transactions

    def get_user_transactions(self, user, transaction_type=None, limit=50) -> List[Dict[str, Any]]:
        """
        Get transactions for a user

        Args:
            user: User ID or wallet address
            transaction_type: Optional transaction type filter
            limit: Maximum number of results

        Returns:
            list: List of transactions
        """
        query = {"user": user}

        if transaction_type:
            query["type"] = transaction_type

        transactions = list(self.collection.find(query).sort("timestamp", -1).limit(limit))
        return transactions

    def get_count(self) -> int:
        """
        Get total number of transactions

        Returns:
            int: Transaction count
        """
        return self.collection.count_documents({})