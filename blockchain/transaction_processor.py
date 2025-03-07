from solana.publickey import PublicKey
from datetime import datetime
import json
import time

from blockchain.solana_client import SolanaClient
from blockchain.token_manager import TokenManager
from utils.logging import get_logger
from config import CONFIG

logger = get_logger(__name__)


class TransactionProcessor:
    """Processes blockchain transactions for betting and rewards"""

    def __init__(self, solana_client=None, token_manager=None):
        self.solana_client = solana_client or SolanaClient()
        self.token_manager = token_manager or TokenManager(self.solana_client)
        self.confirmed_txs = {}  # Cache of confirmed transactions

    def process_bet(self, user_pubkey, amount, prediction_id, metadata=None):
        """
        Process a bet transaction

        Args:
            user_pubkey: User's public key
            amount: Bet amount in PORIX tokens
            prediction_id: ID of the prediction being bet on
            metadata: Additional metadata for the transaction

        Returns:
            dict: Transaction details
        """
        if isinstance(user_pubkey, str):
            user_pubkey = PublicKey(user_pubkey)

        # Validate bet amount
        if amount < CONFIG.MIN_BET_AMOUNT:
            logger.error(f"Bet amount {amount} is below minimum {CONFIG.MIN_BET_AMOUNT}")
            return {
                "success": False,
                "error": f"Bet amount must be at least {CONFIG.MIN_BET_AMOUNT} PORIX"
            }

        if amount > CONFIG.MAX_BET_AMOUNT:
            logger.error(f"Bet amount {amount} exceeds maximum {CONFIG.MAX_BET_AMOUNT}")
            return {
                "success": False,
                "error": f"Bet amount cannot exceed {CONFIG.MAX_BET_AMOUNT} PORIX"
            }

        # Check user's token balance
        user_balance = self.token_manager.get_token_balance(user_pubkey)
        if user_balance < amount:
            logger.error(f"Insufficient balance: {user_balance} < {amount}")
            return {
                "success": False,
                "error": "Insufficient token balance"
            }

        # Create escrow account for the bet
        # In a real implementation, tokens would be transferred to an escrow account
        # Here we'll just burn the tokens to simulate locking them up
        try:
            # Prepare transaction metadata
            tx_metadata = {
                "type": "bet",
                "prediction_id": prediction_id,
                "amount": amount,
                "user": str(user_pubkey),
                "timestamp": datetime.now().isoformat(),
                "custom_metadata": metadata or {}
            }

            # Process the transaction (burn tokens)
            signature = self.token_manager.burn_tokens(user_pubkey, amount)

            if not signature:
                logger.error("Failed to process bet transaction")
                return {
                    "success": False,
                    "error": "Transaction failed"
                }

            # Store transaction details
            tx_details = {
                "signature": signature,
                "user": str(user_pubkey),
                "amount": amount,
                "prediction_id": prediction_id,
                "timestamp": datetime.now().isoformat(),
                "metadata": tx_metadata,
                "status": "confirmed",
                "type": "bet"
            }

            self.confirmed_txs[signature] = tx_details

            logger.info(f"Bet processed successfully: {signature}")
            return {
                "success": True,
                "transaction": tx_details
            }

        except Exception as e:
            logger.error(f"Error processing bet: {str(e)}")
            return {
                "success": False,
                "error": f"Error processing bet: {str(e)}"
            }

    def process_reward(self, user_pubkey, amount, prediction_id, metadata=None):
        """
        Process a reward transaction

        Args:
            user_pubkey: User's public key
            amount: Reward amount in PORIX tokens
            prediction_id: ID of the prediction that was bet on
            metadata: Additional metadata for the transaction

        Returns:
            dict: Transaction details
        """
        if isinstance(user_pubkey, str):
            user_pubkey = PublicKey(user_pubkey)

        try:
            # Prepare transaction metadata
            tx_metadata = {
                "type": "reward",
                "prediction_id": prediction_id,
                "amount": amount,
                "user": str(user_pubkey),
                "timestamp": datetime.now().isoformat(),
                "custom_metadata": metadata or {}
            }

            # Process the transaction (mint new tokens)
            signature = self.token_manager.mint_tokens(user_pubkey, amount)

            if not signature:
                logger.error("Failed to process reward transaction")
                return {
                    "success": False,
                    "error": "Transaction failed"
                }

            # Store transaction details
            tx_details = {
                "signature": signature,
                "user": str(user_pubkey),
                "amount": amount,
                "prediction_id": prediction_id,
                "timestamp": datetime.now().isoformat(),
                "metadata": tx_metadata,
                "status": "confirmed",
                "type": "reward"
            }

            self.confirmed_txs[signature] = tx_details

            logger.info(f"Reward processed successfully: {signature}")
            return {
                "success": True,
                "transaction": tx_details
            }

        except Exception as e:
            logger.error(f"Error processing reward: {str(e)}")
            return {
                "success": False,
                "error": f"Error processing reward: {str(e)}"
            }

    def verify_transaction(self, signature):
        """
        Verify a transaction's status

        Args:
            signature: Transaction signature

        Returns:
            dict: Transaction status
        """
        # Check cache first
        if signature in self.confirmed_txs:
            return {
                "verified": True,
                "status": "confirmed",
                "details": self.confirmed_txs[signature]
            }

        try:
            # Get transaction details from blockchain
            tx_details = self.solana_client.get_transaction(signature)

            if not tx_details:
                logger.error(f"Transaction not found: {signature}")
                return {
                    "verified": False,
                    "status": "not_found",
                    "error": "Transaction not found"
                }

            # Check if confirmed
            if tx_details.get("meta", {}).get("err") is not None:
                return {
                    "verified": False,
                    "status": "failed",
                    "error": "Transaction failed",
                    "details": tx_details
                }

            # Transaction is confirmed
            self.confirmed_txs[signature] = tx_details

            return {
                "verified": True,
                "status": "confirmed",
                "details": tx_details
            }

        except Exception as e:
            logger.error(f"Error verifying transaction: {str(e)}")
            return {
                "verified": False,
                "status": "error",
                "error": f"Error verifying transaction: {str(e)}"
            }

    def get_user_transactions(self, user_pubkey, transaction_type=None, limit=50):
        """
        Get all transactions for a user

        Args:
            user_pubkey: User's public key
            transaction_type: Optional type filter ("bet" or "reward")
            limit: Maximum number of transactions to return

        Returns:
            list: List of transactions
        """
        if isinstance(user_pubkey, str):
            user_pubkey = PublicKey(user_pubkey)

        # In a real implementation, this would query the blockchain or a database
        # Here we'll just filter our cache

        user_txs = []

        for tx in self.confirmed_txs.values():
            if tx.get("user") == str(user_pubkey):
                if not transaction_type or tx.get("type") == transaction_type:
                    user_txs.append(tx)

        # Sort by timestamp (newest first)
        user_txs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        # Apply limit
        return user_txs[:limit]

    def calculate_reward(self, bet_amount, prediction_result):
        """
        Calculate reward amount based on bet amount and prediction result

        Args:
            bet_amount: Original bet amount
            prediction_result: Prediction result object

        Returns:
            float: Reward amount
        """
        # Get base multiplier
        multiplier = CONFIG.REWARD_MULTIPLIER

        # Adjust multiplier based on prediction difficulty
        confidence = prediction_result.get("confidence", 0.5)
        difficulty_factor = 1 + (1 - confidence)  # Higher rewards for less confident predictions

        # Apply multiplier
        return bet_amount * multiplier * difficulty_factor