from functools import wraps
from flask import request, jsonify, current_app
import jwt
from datetime import datetime, timedelta

from utils.logging import get_logger
from config import CONFIG

logger = get_logger(__name__)


def auth_required(f):
    """
    Authentication middleware to validate JWT tokens

    Usage:
        @app.route('/protected')
        @auth_required
        def protected_route():
            return 'Protected content'
    """

    @wraps(f)
    def decorated(*args, **kwargs):
        token = None

        # Get token from Authorization header
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']

            # Check for Bearer token
            if auth_header.startswith('Bearer '):
                token = auth_header.split(' ')[1]

        # Check if token exists
        if not token:
            logger.warning("Authentication failed: No token provided")
            return jsonify({'error': 'Authentication required'}), 401

        try:
            # Verify token
            payload = jwt.decode(
                token,
                CONFIG.JWT_SECRET_KEY,
                algorithms=['HS256']
            )

            # Add user_id to request
            request.user_id = payload['sub']

        except jwt.ExpiredSignatureError:
            logger.warning("Authentication failed: Token expired")
            return jsonify({'error': 'Token expired'}), 401

        except jwt.InvalidTokenError:
            logger.warning("Authentication failed: Invalid token")
            return jsonify({'error': 'Invalid token'}), 401

        return f(*args, **kwargs)

    return decorated


def validate_request(validator):
    """
    Request validation middleware

    Args:
        validator: Validation function that takes request data and returns
                  (is_valid, errors) tuple

    Usage:
        @app.route('/create-user', methods=['POST'])
        @validate_request(validate_user_data)
        def create_user():
            # Request data is valid at this point
            return 'User created'
    """

    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            # Get request data based on content type
            if request.is_json:
                data = request.json
            else:
                data = request.form

            # Validate request data
            is_valid, errors = validator(data)

            if not is_valid:
                logger.warning(f"Request validation failed: {errors}")
                return jsonify({'error': 'Invalid request', 'details': errors}), 400

            return f(*args, **kwargs)

        return decorated

    return decorator


def generate_token(user_id, expires_in=None):
    """
    Generate a JWT token for a user

    Args:
        user_id: User ID to encode in the token
        expires_in: Token expiration in seconds (default from config)

    Returns:
        str: JWT token
    """
    if expires_in is None:
        expires_in = CONFIG.JWT_ACCESS_TOKEN_EXPIRES

    # Set payload
    payload = {
        'sub': user_id,
        'iat': datetime.utcnow(),
        'exp': datetime.utcnow() + timedelta(seconds=expires_in)
    }

    # Generate token
    token = jwt.encode(payload, CONFIG.JWT_SECRET_KEY, algorithm='HS256')

    return token