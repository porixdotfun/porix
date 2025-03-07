from datetime import datetime, timedelta
import uuid
from solana.publickey import PublicKey

from blockchain.transaction_processor import TransactionProcessor
from services.prediction_service import PredictionService
from db.repositories import BetRepository, PredictionRepository
from utils.logging import get_logger
from config import CONFIG

logger = get_logger(__name__)


class BettingService:
    """Service for managing bets on predictions"""

    def __init__(self, bet_repo=None, prediction_repo=None, transaction_processor=None, prediction_service=None):
        self.bet_repo = bet_repo or BetRepository()
        self.prediction_repo = prediction_repo or PredictionRepository()
        self.transaction_processor = transaction_processor or TransactionProcessor()
        self.prediction_service = prediction_service or PredictionService()

    def place_bet(self, user_id, prediction_id, amount, direction=None):
        """
        Place a bet on a prediction

        Args:
            user_id: User ID placing the bet
            prediction_id: ID of the prediction to bet on
            amount: Amount of tokens to bet
            direction: Direction of the bet for price direction predictions ("up" or "down")

        Returns:
            dict: Bet details
        """
        logger.info(f"User {user_id} placing bet of {amount} on prediction {prediction_id}")

        try:
            # Get prediction
            prediction = self.prediction_repo.get_prediction(prediction_id)

            if not prediction:
                logger.error(f"Prediction {prediction_id} not found")
                return {
                    "success": False,
                    "error": "Prediction not found"
                }

            # Check if prediction has expired
            timestamp = datetime.fromisoformat(prediction["timestamp"]) if isinstance(prediction["timestamp"], str) else \
            prediction["timestamp"]
            time_horizon = prediction["time_horizon"]
            expiry = timestamp + self._parse_time_horizon(time_horizon)

            if datetime.now() >= expiry:
                logger.error(f"Prediction {prediction_id} has expired")
                return {
                    "success": False,
                    "error": "Prediction has expired"
                }

            # Validate direction for price direction predictions
            if prediction["prediction_type"] == "price_direction" and direction not in ["up", "down"]:
                logger.error(f"Invalid direction for price direction prediction: {direction}")
                return {
                    "success": False,
                    "error": "Invalid direction for price direction prediction"
                }

            # Generate bet ID
            bet_id = f"bet-{uuid.uuid4()}"

            # Convert direction to expected value
            expected_value = None
            if prediction["prediction_type"] == "price_direction":
                expected_value = 1 if direction == "up" else 0

            # Process transaction
            transaction_result = self.transaction_processor.process_bet(
                user_id, amount, prediction_id,
                metadata={
                    "bet_id": bet_id,
                    "direction": direction
                }
            )

            if not transaction_result.get("success"):
                logger.error(f"Transaction failed: {transaction_result.get('error')}")
                return transaction_result

            # Create bet record
            bet = {
                "bet_id": bet_id,
                "user_id": user_id,
                "prediction_id": prediction_id,
                "amount": amount,
                "timestamp": datetime.now().isoformat(),
                "expected_value": expected_value,
                "direction": direction,
                "status": "placed",
                "transaction_signature": transaction_result["transaction"]["signature"]
            }

            # Store bet
            stored_bet = self.bet_repo.create_bet(bet)

            # Calculate potential reward
            potential_reward = self._calculate_potential_reward(amount, prediction)

            return {
                "success": True,
                "bet": {
                    **stored_bet,
                    "potential_reward": potential_reward
                },
                "transaction": transaction_result["transaction"]
            }

        except Exception as e:
            logger.error(f"Error placing bet: {str(e)}")
            return {
                "success": False,
                "error": f"Error placing bet: {str(e)}"
            }

    def cancel_bet(self, bet_id, user_id):
        """
        Cancel a bet

        Args:
            bet_id: Bet ID to cancel
            user_id: User ID making the request

        Returns:
            dict: Result
        """
        logger.info(f"User {user_id} cancelling bet {bet_id}")

        try:
            # Get bet
            bet = self.bet_repo.get_bet(bet_id)

            if not bet:
                logger.error(f"Bet {bet_id} not found")
                return {
                    "success": False,
                    "error": "Bet not found"
                }

            # Check if user owns the bet
            if bet["user_id"] != user_id:
                logger.error(f"User {user_id} does not own bet {bet_id}")
                return {
                    "success": False,
                    "error": "Unauthorized to cancel this bet"
                }

            # Check if bet can be cancelled (not settled or already cancelled)
            if bet["status"] != "placed":
                logger.error(f"Bet {bet_id} cannot be cancelled (status: {bet['status']})")
                return {
                    "success": False,
                    "error": f"Bet cannot be cancelled (status: {bet['status']})"
                }

            # Get prediction
            prediction_id = bet["prediction_id"]
            prediction = self.prediction_repo.get_prediction(prediction_id)

            if not prediction:
                logger.error(f"Prediction {prediction_id} not found")
                return {
                    "success": False,
                    "error": "Prediction not found"
                }

            # Check if prediction has expired
            timestamp = datetime.fromisoformat(prediction["timestamp"]) if isinstance(prediction["timestamp"], str) else \
            prediction["timestamp"]
            time_horizon = prediction["time_horizon"]
            expiry = timestamp + self._parse_time_horizon(time_horizon)

            if datetime.now() >= expiry:
                logger.error(f"Bet {bet_id} cannot be cancelled (prediction has expired)")
                return {
                    "success": False,
                    "error": "Bet cannot be cancelled (prediction has expired)"
                }

            # Process refund transaction
            # In a real implementation, this would refund the tokens from escrow
            # Here we'll mint new tokens to simulate the refund
            amount = bet["amount"]
            refund_result = self.transaction_processor.process_reward(
                user_id, amount, prediction_id,
                metadata={
                    "bet_id": bet_id,
                    "type": "refund"
                }
            )

            if not refund_result.get("success"):
                logger.error(f"Refund transaction failed: {refund_result.get('error')}")
                return refund_result

            # Update bet status
            self.bet_repo.update_bet(bet_id, {
                "status": "cancelled",
                "refund_transaction_signature": refund_result["transaction"]["signature"],
                "cancelled_at": datetime.now().isoformat()
            })

            return {
                "success": True,
                "message": "Bet cancelled successfully",
                "transaction": refund_result["transaction"]
            }

        except Exception as e:
            logger.error(f"Error cancelling bet: {str(e)}")
            return {
                "success": False,
                "error": f"Error cancelling bet: {str(e)}"
            }

    def settle_bet(self, bet_id):
        """
        Settle a bet based on prediction outcome

        Args:
            bet_id: Bet ID to settle

        Returns:
            dict: Settlement result
        """
        logger.info(f"Settling bet {bet_id}")

        try:
            # Get bet
            bet = self.bet_repo.get_bet(bet_id)

            if not bet:
                logger.error(f"Bet {bet_id} not found")
                return {
                    "success": False,
                    "error": "Bet not found"
                }

            # Check if bet is already settled
            if bet["status"] in ["settled", "cancelled"]:
                logger.error(f"Bet {bet_id} is already {bet['status']}")
                return {
                    "success": False,
                    "error": f"Bet is already {bet['status']}"
                }

            # Get prediction outcome
            prediction_id = bet["prediction_id"]
            outcome = self.prediction_service.get_prediction_outcome(prediction_id)

            if not outcome or outcome["status"] == "pending":
                logger.error(f"Prediction {prediction_id} outcome not available yet")
                return {
                    "success": False,
                    "error": "Prediction outcome not available yet"
                }

            # Determine if bet won
            won = self._did_bet_win(bet, outcome)

            # Process reward if bet won
            user_id = bet["user_id"]
            amount = bet["amount"]

            if won:
                # Calculate reward
                reward_amount = self._calculate_reward(amount, outcome)

                # Process reward transaction
                reward_result = self.transaction_processor.process_reward(
                    user_id, reward_amount, prediction_id,
                    metadata={
                        "bet_id": bet_id,
                        "type": "reward"
                    }
                )

                if not reward_result.get("success"):
                    logger.error(f"Reward transaction failed: {reward_result.get('error')}")
                    return reward_result

                # Update bet status
                self.bet_repo.update_bet(bet_id, {
                    "status": "settled",
                    "outcome": "won",
                    "reward_amount": reward_amount,
                    "reward_transaction_signature": reward_result["transaction"]["signature"],
                    "settled_at": datetime.now().isoformat()
                })

                return {
                    "success": True,
                    "result": "won",
                    "reward_amount": reward_amount,
                    "transaction": reward_result["transaction"]
                }

            else:
                # Update bet status for loss
                self.bet_repo.update_bet(bet_id, {
                    "status": "settled",
                    "outcome": "lost",
                    "settled_at": datetime.now().isoformat()
                })

                return {
                    "success": True,
                    "result": "lost",
                    "message": "Bet settled as a loss"
                }

        except Exception as e:
            logger.error(f"Error settling bet: {str(e)}")
            return {
                "success": False,
                "error": f"Error settling bet: {str(e)}"
            }

    def settle_expired_bets(self):
        """
        Settle all expired bets

        Returns:
            dict: Settlement results
        """
        logger.info("Settling expired bets")

        try:
            # Get all placed bets
            placed_bets = self.bet_repo.get_bets_by_status("placed")

            results = {
                "processed": 0,
                "settled": 0,
                "errors": 0,
                "details": []
            }

            for bet in placed_bets:
                bet_id = bet["bet_id"]
                prediction_id = bet["prediction_id"]

                # Get prediction
                prediction = self.prediction_repo.get_prediction(prediction_id)

                if not prediction:
                    logger.error(f"Prediction {prediction_id} not found for bet {bet_id}")
                    results["errors"] += 1
                    results["details"].append({
                        "bet_id": bet_id,
                        "error": "Prediction not found"
                    })
                    continue

                # Check if prediction has expired
                timestamp = datetime.fromisoformat(prediction["timestamp"]) if isinstance(prediction["timestamp"],
                                                                                          str) else prediction[
                    "timestamp"]
                time_horizon = prediction["time_horizon"]
                expiry = timestamp + self._parse_time_horizon(time_horizon)

                if datetime.now() < expiry:
                    # Prediction hasn't expired yet, skip
                    continue

                # Settle the bet
                result = self.settle_bet(bet_id)
                results["processed"] += 1

                if result.get("success"):
                    results["settled"] += 1
                    results["details"].append({
                        "bet_id": bet_id,
                        "result": result.get("result")
                    })
                else:
                    results["errors"] += 1
                    results["details"].append({
                        "bet_id": bet_id,
                        "error": result.get("error")
                    })

            logger.info(f"Settled {results['settled']} bets with {results['errors']} errors")
            return {
                "success": True,
                "results": results
            }

        except Exception as e:
            logger.error(f"Error settling expired bets: {str(e)}")
            return {
                "success": False,
                "error": f"Error settling expired bets: {str(e)}"
            }

    def get_bet(self, bet_id):
        """
        Get a bet by ID

        Args:
            bet_id: Bet ID

        Returns:
            dict: Bet details
        """
        return self.bet_repo.get_bet(bet_id)

    def get_betting_history(self, user_id=None, prediction_id=None, status=None, limit=50):
        """
        Get betting history

        Args:
            user_id: Filter by user ID
            prediction_id: Filter by prediction ID
            status: Filter by status
            limit: Maximum number of results

        Returns:
            list: List of bets
        """
        filters = {}

        if user_id:
            filters["user_id"] = user_id

        if prediction_id:
            filters["prediction_id"] = prediction_id

        if status:
            filters["status"] = status

        bets = self.bet_repo.get_bets(filters, limit)

        return bets

    def get_bet_count(self):
        """
        Get total number of bets

        Returns:
            int: Total bet count
        """
        return self.bet_repo.get_count()

    def get_leaderboard(self, time_period="all_time", prediction_type=None, limit=50):
        """
        Get betting leaderboard

        Args:
            time_period: Time period filter ("all_time", "month", "week", "day")
            prediction_type: Filter by prediction type
            limit: Maximum number of results

        Returns:
            list: Leaderboard entries
        """
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

        # Get all settled bets within time period
        filters = {"status": "settled"}

        if start_date:
            filters["timestamp_after"] = start_date.isoformat()

        bets = self.bet_repo.get_bets(filters, 10000)  # Get a large number to calculate stats

        if prediction_type:
            # Filter by prediction type
            bets = [b for b in bets if self._get_prediction_type(b["prediction_id"]) == prediction_type]

        # Group by user and calculate stats
        user_stats = {}

        for bet in bets:
            user_id = bet["user_id"]

            if user_id not in user_stats:
                user_stats[user_id] = {
                    "user_id": user_id,
                    "total_bets": 0,
                    "wins": 0,
                    "losses": 0,
                    "total_amount_bet": 0,
                    "total_rewards": 0,
                    "net_profit": 0,
                    "win_rate": 0
                }

            stats = user_stats[user_id]
            stats["total_bets"] += 1
            stats["total_amount_bet"] += bet["amount"]

            if bet["outcome"] == "won":
                stats["wins"] += 1
                stats["total_rewards"] += bet.get("reward_amount", 0)
            else:
                stats["losses"] += 1

            stats["net_profit"] = stats["total_rewards"] - stats["total_amount_bet"]
            stats["win_rate"] = stats["wins"] / stats["total_bets"] * 100

        # Convert to list and sort by net profit
        leaderboard = list(user_stats.values())
        leaderboard.sort(key=lambda x: x["net_profit"], reverse=True)

        # Apply limit
        return leaderboard[:limit]

    def _did_bet_win(self, bet, outcome):
        """
        Determine if a bet won based on the prediction outcome

        Args:
            bet: Bet details
            outcome: Prediction outcome

        Returns:
            bool: Whether the bet won
        """
        # Get prediction type
        prediction_id = bet["prediction_id"]
        prediction = self.prediction_repo.get_prediction(prediction_id)

        if not prediction:
            logger.error(f"Prediction {prediction_id} not found")
            return False

        prediction_type = prediction["prediction_type"]

        if prediction_type == "price_direction":
            # For price direction, check if the bet direction matches the outcome
            direction = bet["direction"]
            actual_direction = "up" if outcome["actual_value"] == 1 else "down"

            return direction == actual_direction

        else:
            # For other prediction types, check if the prediction was correct
            return outcome["is_correct"]

    def _calculate_reward(self, amount, outcome):
        """
        Calculate reward for a winning bet

        Args:
            amount: Bet amount
            outcome: Prediction outcome

        Returns:
            float: Reward amount
        """
        # Base multiplier
        multiplier = CONFIG.REWARD_MULTIPLIER

        # For price direction bets, adjust based on probability
        if "probability_up" in outcome:
            prob_up = outcome["probability_up"]
            prob_down = 1 - prob_up

            # Adjust multiplier based on predicted probability
            # Lower probability outcomes have higher rewards
            if outcome["actual_value"] == 1:  # Up
                difficulty_factor = 1 + (1 - prob_up)
            else:  # Down
                difficulty_factor = 1 + (1 - prob_down)

            multiplier *= difficulty_factor

        # Calculate reward
        reward = amount * multiplier

        return reward

    def _calculate_potential_reward(self, amount, prediction):
        """
        Calculate potential reward for a bet

        Args:
            amount: Bet amount
            prediction: Prediction details

        Returns:
            float: Potential reward amount
        """
        # Base multiplier
        multiplier = CONFIG.REWARD_MULTIPLIER

        # For price direction bets, adjust based on probability
        if prediction["prediction_type"] == "price_direction":
            prob_up = prediction.get("probability_up", 0.5)
            prob_down = 1 - prob_up

            # Take the lower probability for the potential reward calculation
            # (assumes user bets on the less likely outcome)
            difficulty_factor = 1 + (1 - min(prob_up, prob_down))
            multiplier *= difficulty_factor

        # Calculate potential reward
        potential_reward = amount * multiplier

        return potential_reward

    def _get_prediction_type(self, prediction_id):
        """
        Get prediction type for a prediction ID

        Args:
            prediction_id: Prediction ID

        Returns:
            str: Prediction type
        """
        prediction = self.prediction_repo.get_prediction(prediction_id)

        if not prediction:
            return None

        return prediction["prediction_type"]

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