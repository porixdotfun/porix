import hashlib
import base64
import jwt
from datetime import datetime, timedelta
from solana.publickey import PublicKey

from blockchain.solana_client import SolanaClient
from blockchain.token_manager import TokenManager
from db.repositories import UserRepository
from api.middleware import generate_token
from utils.logging import get_logger
from utils.security import verify_signature
from config import CONFIG

logger = get_logger(__name__)


class UserService:
    """Service for managing users"""

    def __init__(self, user_repo=None, solana_client=None, token_manager=None):
        self.user_repo = user_repo or UserRepository()
        self.solana_client = solana_client or SolanaClient()
        self.token_manager = token_manager or TokenManager(self.solana_client)

    def register_user(self, username, wallet_address, email=None):
        """
        Register a new user

        Args:
            username: User's chosen username
            wallet_address: User's Solana wallet address
            email: Optional email address

        Returns:
            dict: Registration result
        """
        logger.info(f"Registering new user {username} with wallet {wallet_address}")

        try:
            # Check if username is already taken
            existing_user = self.user_repo.get_user_by_username(username)

            if existing_user:
                logger.error(f"Username {username} is already taken")
                return {
                    "success": False,
                    "error": "Username is already taken"
                }

            # Check if wallet address is already registered
            existing_wallet = self.user_repo.get_user_by_wallet(wallet_address)

            if existing_wallet:
                logger.error(f"Wallet address {wallet_address} is already registered")
                return {
                    "success": False,
                    "error": "Wallet address is already registered"
                }

            # Generate user ID
            user_id = self._generate_user_id(wallet_address)

            # Create user record
            user = {
                "user_id": user_id,
                "username": username,
                "wallet_address": wallet_address,
                "email": email,
                "created_at": datetime.now().isoformat(),
                "last_login": None,
                "status": "active"
            }

            # Store user
            stored_user = self.user_repo.create_user(user)

            # Create token account for the user
            token_account = self.token_manager.create_token_account(wallet_address)

            # Mint welcome tokens
            welcome_amount = 10  # Welcome tokens
            mint_result = self.token_manager.mint_tokens(wallet_address, welcome_amount)

            # Generate authentication token
            auth_token = generate_token(user_id)

            return {
                "success": True,
                "user": {
                    "user_id": user_id,
                    "username": username,
                    "wallet_address": wallet_address,
                    "token_balance": welcome_amount,
                    "created_at": user["created_at"]
                },
                "auth_token": auth_token,
                "token_account": str(token_account) if token_account else None,
                "welcome_tokens": welcome_amount,
                "mint_transaction": mint_result
            }

        except Exception as e:
            logger.error(f"Error registering user: {str(e)}")
            return {
                "success": False,
                "error": f"Error registering user: {str(e)}"
            }

    def login_user(self, wallet_address, signature, message):
        """
        Log in a user using wallet signature

        Args:
            wallet_address: User's wallet address
            signature: Signature from wallet
            message: Message that was signed

        Returns:
            dict: Login result
        """
        logger.info(f"User login attempt with wallet {wallet_address}")

        try:
            # Verify signature
            if not verify_signature(wallet_address, signature, message):
                logger.error(f"Invalid signature for wallet {wallet_address}")
                return {
                    "success": False,
                    "error": "Invalid signature"
                }

            # Get user by wallet address
            user = self.user_repo.get_user_by_wallet(wallet_address)

            if not user:
                logger.error(f"User not found for wallet {wallet_address}")
                return {
                    "success": False,
                    "error": "User not found"
                }

            # Update last login
            self.user_repo.update_user(user["user_id"], {
                "last_login": datetime.now().isoformat()
            })

            # Generate authentication token
            auth_token = generate_token(user["user_id"])

            # Get token balance
            token_balance = self.token_manager.get_token_balance(wallet_address)

            return {
                "success": True,
                "user": {
                    "user_id": user["user_id"],
                    "username": user["username"],
                    "wallet_address": wallet_address,
                    "token_balance": token_balance,
                    "created_at": user["created_at"],
                    "last_login": datetime.now().isoformat()
                },
                "auth_token": auth_token
            }

        except Exception as e:
            logger.error(f"Error logging in user: {str(e)}")
            return {
                "success": False,
                "error": f"Error logging in user: {str(e)}"
            }

    def get_user(self, user_id):
        """
        Get user by ID

        Args:
            user_id: User ID

        Returns:
            dict: User details
        """
        user = self.user_repo.get_user(user_id)

        if not user:
            return None

        # Get token balance
        token_balance = self.token_manager.get_token_balance(user["wallet_address"])

        # Add balance to user details
        user["token_balance"] = token_balance

        return user

    def get_balance(self, user_id):
        """
        Get user's token balance

        Args:
            user_id: User ID

        Returns:
            float: Token balance
        """
        user = self.user_repo.get_user(user_id)

        if not user:
            return None

        return self.token_manager.get_token_balance(user["wallet_address"])

    def get_user_stats(self, user_id):
        """
        Get user statistics

        Args:
            user_id: User ID

        Returns:
            dict: User statistics
        """
        user = self.user_repo.get_user(user_id)

        if not user:
            return None

        # Get user's bets and predictions
        from db.repositories import BetRepository, PredictionRepository
        bet_repo = BetRepository()
        prediction_repo = PredictionRepository()

        bets = bet_repo.get_bets({"user_id": user_id}, 1000)

        # Calculate betting stats
        total_bets = len(bets)
        active_bets = sum(1 for b in bets if b["status"] == "placed")
        won_bets = sum(1 for b in bets if b.get("outcome") == "won")
        lost_bets = sum(1 for b in bets if b.get("outcome") == "lost")
        win_rate = won_bets / total_bets * 100 if total_bets > 0 else 0

        # Calculate amount stats
        total_bet_amount = sum(b["amount"] for b in bets)
        total_rewards = sum(b.get("reward_amount", 0) for b in bets if b.get("outcome") == "won")
        net_profit = total_rewards - total_bet_amount

        # Get token balance
        token_balance = self.token_manager.get_token_balance(user["wallet_address"])

        return {
            "user_id": user_id,
            "username": user["username"],
            "wallet_address": user["wallet_address"],
            "token_balance": token_balance,
            "created_at": user["created_at"],
            "last_login": user.get("last_login"),
            "betting_stats": {
                "total_bets": total_bets,
                "active_bets": active_bets,
                "won_bets": won_bets,
                "lost_bets": lost_bets,
                "win_rate": win_rate,
                "total_bet_amount": total_bet_amount,
                "total_rewards": total_rewards,
                "net_profit": net_profit
            }
        }

    def get_user_count(self):
        """
        Get total number of users

        Returns:
            int: Total user count
        """
        return self.user_repo.get_count()

    def update_user(self, user_id, updates):
        """
        Update user details

        Args:
            user_id: User ID
            updates: Fields to update

        Returns:
            dict: Updated user
        """
        # Verify user exists
        user = self.user_repo.get_user(user_id)

        if not user:
            logger.error(f"User {user_id} not found")
            return None

        # Apply updates
        updated_user = self.user_repo.update_user(user_id, updates)

        return updated_user

    def _generate_user_id(self, wallet_address):
        """
        Generate a user ID based on wallet address

        Args:
            wallet_address: User's wallet address

        Returns:
            str: User ID
        """
        # Create a hash of the wallet address
        hash_input = f"{wallet_address}:{datetime.now().isoformat()}"
        hash_bytes = hashlib.sha256(hash_input.encode()).digest()

        # Use the first 16 bytes as a user ID
        user_id = base64.urlsafe_b64encode(hash_bytes[:16]).decode().rstrip("=")

        return f"user_{user_id}"