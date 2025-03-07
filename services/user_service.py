from datetime import datetime, timedelta

from blockchain.nft_manager import NFTManager
from db.repositories import BetRepository, UserRepository
from utils.logging import get_logger
from config import CONFIG

logger = get_logger(__name__)


class ReputationService:
    """Service for managing user reputation and NFTs"""

    def __init__(self, nft_manager=None, bet_repo=None, user_repo=None):
        self.nft_manager = nft_manager or NFTManager()
        self.bet_repo = bet_repo or BetRepository()
        self.user_repo = user_repo or UserRepository()

    def mint_reputation_nft(self, user_id):
        """
        Mint a reputation NFT for a user based on their prediction history

        Args:
            user_id: User ID

        Returns:
            dict: Minting result
        """
        logger.info(f"Minting reputation NFT for user {user_id}")

        try:
            # Get user
            user = self.user_repo.get_user(user_id)

            if not user:
                logger.error(f"User {user_id} not found")
                return {
                    "success": False,
                    "error": "User not found"
                }

            # Get user's wallet address
            wallet_address = user["wallet_address"]

            # Get user's betting history
            bets = self.bet_repo.get_bets({"user_id": user_id}, 1000)

            # Filter to settled bets
            settled_bets = [b for b in bets if b["status"] == "settled"]

            if not settled_bets:
                logger.error(f"User {user_id} has no settled bets")
                return {
                    "success": False,
                    "error": "User has no settled bets to generate reputation"
                }

            # Get prediction results
            from db.repositories import PredictionRepository
            prediction_repo = PredictionRepository()

            prediction_results = []
            for bet in settled_bets:
                prediction_id = bet["prediction_id"]
                prediction = prediction_repo.get_prediction(prediction_id)

                if prediction:
                    prediction_results.append({
                        "prediction_id": prediction_id,
                        "asset": prediction.get("asset", prediction.get("collection")),
                        "prediction_type": prediction["prediction_type"],
                        "predicted_value": prediction["predicted_value"],
                        "actual_value": prediction.get("actual_value"),
                        "timestamp": prediction["timestamp"],
                        "is_correct": bet["outcome"] == "won"
                    })

            # Mint NFT
            mint_address, tx_signature = self.nft_manager.create_prediction_nft(
                wallet_address, prediction_results
            )

            if not mint_address:
                logger.error("Failed to mint reputation NFT")
                return {
                    "success": False,
                    "error": "Failed to mint NFT"
                }

            # Get NFT metadata
            metadata = self.nft_manager.get_nft_metadata(mint_address)

            # Store NFT record in user's account
            user_nfts = user.get("reputation_nfts", [])
            user_nfts.append({
                "mint_address": mint_address,
                "mint_transaction": tx_signature,
                "created_at": datetime.now().isoformat(),
                "prediction_count": len(prediction_results),
                "accuracy": metadata.get("attributes", {}).get("Accuracy", "0%")
            })

            self.user_repo.update_user(user_id, {
                "reputation_nfts": user_nfts
            })

            return {
                "success": True,
                "nft": {
                    "mint_address": mint_address,
                    "transaction": tx_signature,
                    "metadata": metadata
                }
            }

        except Exception as e:
            logger.error(f"Error minting reputation NFT: {str(e)}")
            return {
                "success": False,
                "error": f"Error minting reputation NFT: {str(e)}"
            }

    def get_user_nfts(self, user_id):
        """
        Get a user's reputation NFTs

        Args:
            user_id: User ID

        Returns:
            list: List of NFTs
        """
        # Get user
        user = self.user_repo.get_user(user_id)

        if not user:
            logger.error(f"User {user_id} not found")
            return []

        # Get user's wallet address
        wallet_address = user["wallet_address"]

        # Get NFTs from blockchain
        nfts = self.nft_manager.get_user_nfts(wallet_address)

        return nfts

    def get_nft_count(self):
        """
        Get total number of reputation NFTs

        Returns:
            int: Total NFT count
        """
        # In a real implementation, this would query a database
        # Here we'll just count NFTs across all users
        users = self.user_repo.get_users({}, 1000)

        total_nfts = sum(len(user.get("reputation_nfts", [])) for user in users)

        return total_nfts

    def get_leaderboard(self, time_period="all_time", limit=50):
        """
        Get reputation leaderboard

        Args:
            time_period: Time period filter ("all_time", "month", "week", "day")
            limit: Maximum number of results

        Returns:
            list: Leaderboard entries
        """
        # Get all users
        users = self.user_repo.get_users({}, 1000)

        # Calculate date range based on time period
        end_date = datetime.now()

        if time_period == "month":
            start_date = end_date - timedelta(days=30)
        elif time_period == "week":
            start_date = end_date - timedelta(days=7)
        elif time_period == "day":
            start_date = end_date - timedelta(days=1)
        else:
            # All time
            start_date = None

        # Get settled bets for each user
        bet_repo = BetRepository()

        leaderboard = []
        for user in users:
            user_id = user["user_id"]
            username = user["username"]

            # Get user's bets
            filters = {"user_id": user_id, "status": "settled"}

            if start_date:
                filters["timestamp_after"] = start_date.isoformat()

            bets = bet_repo.get_bets(filters, 1000)

            if not bets:
                continue

            # Calculate reputation stats
            total_bets = len(bets)
            won_bets = sum(1 for b in bets if b.get("outcome") == "won")
            accuracy = won_bets / total_bets if total_bets > 0 else 0

            # Calculate reputation score
            # Formula: (wins * 10) + (accuracy * 100) + (total_bets * 0.5)
            reputation_score = (won_bets * 10) + (accuracy * 100) + (total_bets * 0.5)

            # Determine rank based on accuracy and total predictions
            rank = self._calculate_rank(accuracy, total_bets)

            leaderboard.append({
                "user_id": user_id,
                "username": username,
                "total_predictions": total_bets,
                "correct_predictions": won_bets,
                "accuracy": accuracy * 100,  # as percentage
                "reputation_score": reputation_score,
                "rank": rank,
                "nft_count": len(user.get("reputation_nfts", []))
            })

        # Sort by reputation score
        leaderboard.sort(key=lambda x: x["reputation_score"], reverse=True)

        # Apply limit
        return leaderboard[:limit]

    def _calculate_rank(self, accuracy, total_predictions):
        """
        Calculate user rank based on performance

        Args:
            accuracy: Prediction accuracy (0-1)
            total_predictions: Total number of predictions

        Returns:
            str: Rank title
        """
        if total_predictions < 5:
            return "Novice"
        elif accuracy < 0.5:
            return "Apprentice"
        elif accuracy < 0.6:
            return "Adept"
        elif accuracy < 0.7:
            return "Expert"
        elif accuracy < 0.8:
            return "Master"
        else:
            return "Oracle"