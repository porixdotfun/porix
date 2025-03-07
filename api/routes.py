from flask import Blueprint, request, jsonify, current_app
from flask_cors import cross_origin
import json
from datetime import datetime
import uuid

from services.prediction_service import PredictionService
from services.betting_service import BettingService
from services.user_service import UserService
from services.reputation_service import ReputationService
from api.middleware import auth_required, validate_request
from api.validators import validate_bet, validate_prediction_request
from utils.logging import get_logger
from config import CONFIG

logger = get_logger(__name__)

# Create blueprints
api_bp = Blueprint("api", __name__, url_prefix=CONFIG.API_PREFIX)
prediction_bp = Blueprint("prediction", __name__, url_prefix="/prediction")
betting_bp = Blueprint("betting", __name__, url_prefix="/betting")
user_bp = Blueprint("user", __name__, url_prefix="/user")
reputation_bp = Blueprint("reputation", __name__, url_prefix="/reputation")

# Initialize services
prediction_service = PredictionService()
betting_service = BettingService()
user_service = UserService()
reputation_service = ReputationService()


# Register error handlers
@api_bp.errorhandler(400)
def handle_bad_request(error):
    return jsonify({"error": "Bad request", "message": str(error)}), 400


@api_bp.errorhandler(401)
def handle_unauthorized(error):
    return jsonify({"error": "Unauthorized", "message": str(error)}), 401


@api_bp.errorhandler(404)
def handle_not_found(error):
    return jsonify({"error": "Not found", "message": str(error)}), 404


@api_bp.errorhandler(500)
def handle_server_error(error):
    logger.error(f"Server error: {str(error)}")
    return jsonify({"error": "Server error", "message": "An internal server error occurred"}), 500


# Prediction routes
@prediction_bp.route("/price-direction", methods=["POST"])
@auth_required
@validate_request(validate_prediction_request)
def predict_price_direction():
    """Get a price direction prediction"""
    data = request.json
    asset = data.get("asset")
    time_horizon = data.get("time_horizon", "24h")

    # Get prediction
    prediction = prediction_service.predict_price_direction(asset, time_horizon)

    return jsonify(prediction)


@prediction_bp.route("/price-target", methods=["POST"])
@auth_required
@validate_request(validate_prediction_request)
def predict_price_target():
    """Get a price target prediction"""
    data = request.json
    asset = data.get("asset")
    time_horizon = data.get("time_horizon", "24h")

    # Get prediction
    prediction = prediction_service.predict_price_target(asset, time_horizon)

    return jsonify(prediction)


@prediction_bp.route("/nft-floor", methods=["POST"])
@auth_required
@validate_request(validate_prediction_request)
def predict_nft_floor():
    """Get an NFT floor price prediction"""
    data = request.json
    collection = data.get("collection")
    time_horizon = data.get("time_horizon", "24h")

    # Get prediction
    prediction = prediction_service.predict_nft_floor_price(collection, time_horizon)

    return jsonify(prediction)


@prediction_bp.route("/history", methods=["GET"])
@auth_required
def get_prediction_history():
    """Get prediction history"""
    user_id = request.args.get("user_id")
    asset = request.args.get("asset")
    prediction_type = request.args.get("type")
    limit = int(request.args.get("limit", 50))

    # Get prediction history
    history = prediction_service.get_prediction_history(
        user_id=user_id,
        asset=asset,
        prediction_type=prediction_type,
        limit=limit
    )

    return jsonify(history)


@prediction_bp.route("/<prediction_id>", methods=["GET"])
@auth_required
def get_prediction(prediction_id):
    """Get a specific prediction"""
    prediction = prediction_service.get_prediction(prediction_id)

    if not prediction:
        return jsonify({"error": "Prediction not found"}), 404

    return jsonify(prediction)


@prediction_bp.route("/<prediction_id>/outcome", methods=["GET"])
@auth_required
def get_prediction_outcome(prediction_id):
    """Get the outcome of a prediction"""
    outcome = prediction_service.get_prediction_outcome(prediction_id)

    if not outcome:
        return jsonify({"error": "Prediction outcome not found"}), 404

    return jsonify(outcome)


# Betting routes
@betting_bp.route("/place", methods=["POST"])
@auth_required
@validate_request(validate_bet)
def place_bet():
    """Place a bet on a prediction"""
    data = request.json
    user_id = data.get("user_id")
    prediction_id = data.get("prediction_id")
    amount = data.get("amount")
    direction = data.get("direction")  # "up" or "down" for direction bets

    # Place bet
    result = betting_service.place_bet(user_id, prediction_id, amount, direction)

    if not result.get("success"):
        return jsonify(result), 400

    return jsonify(result)


@betting_bp.route("/history", methods=["GET"])
@auth_required
def get_betting_history():
    """Get betting history"""
    user_id = request.args.get("user_id")
    prediction_id = request.args.get("prediction_id")
    status = request.args.get("status")
    limit = int(request.args.get("limit", 50))

    # Get betting history
    history = betting_service.get_betting_history(
        user_id=user_id,
        prediction_id=prediction_id,
        status=status,
        limit=limit
    )

    return jsonify(history)


@betting_bp.route("/<bet_id>", methods=["GET"])
@auth_required
def get_bet(bet_id):
    """Get a specific bet"""
    bet = betting_service.get_bet(bet_id)

    if not bet:
        return jsonify({"error": "Bet not found"}), 404

    return jsonify(bet)


@betting_bp.route("/<bet_id>/cancel", methods=["POST"])
@auth_required
def cancel_bet(bet_id):
    """Cancel a bet"""
    user_id = request.json.get("user_id")

    # Cancel bet
    result = betting_service.cancel_bet(bet_id, user_id)

    if not result.get("success"):
        return jsonify(result), 400

    return jsonify(result)


@betting_bp.route("/leaderboard", methods=["GET"])
def get_leaderboard():
    """Get betting leaderboard"""
    time_period = request.args.get("time_period", "all_time")
    prediction_type = request.args.get("prediction_type")
    limit = int(request.args.get("limit", 50))

    # Get leaderboard
    leaderboard = betting_service.get_leaderboard(
        time_period=time_period,
        prediction_type=prediction_type,
        limit=limit
    )

    return jsonify(leaderboard)


# User routes
@user_bp.route("/register", methods=["POST"])
def register_user():
    """Register a new user"""
    data = request.json
    username = data.get("username")
    wallet_address = data.get("wallet_address")
    email = data.get("email")

    # Register user
    result = user_service.register_user(username, wallet_address, email)

    if not result.get("success"):
        return jsonify(result), 400

    return jsonify(result)


@user_bp.route("/login", methods=["POST"])
def login_user():
    """Log in a user"""
    data = request.json
    wallet_address = data.get("wallet_address")
    signature = data.get("signature")
    message = data.get("message")

    # Log in user
    result = user_service.login_user(wallet_address, signature, message)

    if not result.get("success"):
        return jsonify(result), 401

    return jsonify(result)


@user_bp.route("/<user_id>", methods=["GET"])
@auth_required
def get_user(user_id):
    """Get user details"""
    user = user_service.get_user(user_id)

    if not user:
        return jsonify({"error": "User not found"}), 404

    return jsonify(user)


@user_bp.route("/<user_id>/balance", methods=["GET"])
@auth_required
def get_balance(user_id):
    """Get user balance"""
    balance = user_service.get_balance(user_id)

    if balance is None:
        return jsonify({"error": "User not found"}), 404

    return jsonify({"user_id": user_id, "balance": balance})


@user_bp.route("/<user_id>/stats", methods=["GET"])
@auth_required
def get_user_stats(user_id):
    """Get user stats"""
    stats = user_service.get_user_stats(user_id)

    if not stats:
        return jsonify({"error": "User not found"}), 404

    return jsonify(stats)


# Reputation routes
@reputation_bp.route("/mint-nft", methods=["POST"])
@auth_required
def mint_reputation_nft():
    """Mint a reputation NFT for a user"""
    data = request.json
    user_id = data.get("user_id")

    # Mint NFT
    result = reputation_service.mint_reputation_nft(user_id)

    if not result.get("success"):
        return jsonify(result), 400

    return jsonify(result)


@reputation_bp.route("/<user_id>/nfts", methods=["GET"])
@auth_required
def get_user_nfts(user_id):
    """Get a user's reputation NFTs"""
    nfts = reputation_service.get_user_nfts(user_id)

    return jsonify(nfts)


@reputation_bp.route("/leaderboard", methods=["GET"])
def get_reputation_leaderboard():
    """Get reputation leaderboard"""
    time_period = request.args.get("time_period", "all_time")
    limit = int(request.args.get("limit", 50))

    # Get leaderboard
    leaderboard = reputation_service.get_leaderboard(
        time_period=time_period,
        limit=limit
    )

    return jsonify(leaderboard)


# Main routes
@api_bp.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "version": CONFIG.APP_VERSION,
        "timestamp": datetime.now().isoformat()
    })


@api_bp.route("/stats", methods=["GET"])
def get_platform_stats():
    """Get platform statistics"""
    stats = {
        "total_users": user_service.get_user_count(),
        "total_predictions": prediction_service.get_prediction_count(),
        "total_bets": betting_service.get_bet_count(),
        "reputation_nfts": reputation_service.get_nft_count(),
        "updated_at": datetime.now().isoformat()
    }

    return jsonify(stats)


# Register blueprints
def register_routes(app):
    """Register all API routes"""
    api_bp.register_blueprint(prediction_bp)
    api_bp.register_blueprint(betting_bp)
    api_bp.register_blueprint(user_bp)
    api_bp.register_blueprint(reputation_bp)

    app.register_blueprint(api_bp)

    logger.info("API routes registered")

    return app