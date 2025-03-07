from datetime import datetime


def validate_prediction_request(data):
    """
    Validate a prediction request

    Args:
        data: Request data

    Returns:
        tuple: (is_valid, errors)
    """
    errors = {}

    # Check required fields
    if 'asset' not in data and 'collection' not in data:
        errors['asset'] = 'Either asset or collection is required'

    # Validate time horizon
    if 'time_horizon' in data:
        time_horizon = data['time_horizon']
        if not isinstance(time_horizon, str):
            errors['time_horizon'] = 'Time horizon must be a string'
        elif not (time_horizon.endswith('h') or time_horizon.endswith('d')):
            errors['time_horizon'] = 'Time horizon must end with h (hours) or d (days)'
        else:
            try:
                value = int(time_horizon[:-1])
                if value <= 0:
                    errors['time_horizon'] = 'Time horizon must be positive'
            except ValueError:
                errors['time_horizon'] = 'Invalid time horizon format'

    return len(errors) == 0, errors


def validate_bet(data):
    """
    Validate a bet request

    Args:
        data: Request data

    Returns:
        tuple: (is_valid, errors)
    """
    errors = {}

    # Check required fields
    required_fields = ['user_id', 'prediction_id', 'amount']
    for field in required_fields:
        if field not in data:
            errors[field] = f'Field {field} is required'

    # Validate amount
    if 'amount' in data:
        try:
            amount = float(data['amount'])
            if amount <= 0:
                errors['amount'] = 'Amount must be positive'
        except (ValueError, TypeError):
            errors['amount'] = 'Amount must be a number'

    # Validate direction for direction bets
    if 'direction' in data:
        direction = data['direction']
        if direction not in ['up', 'down']:
            errors['direction'] = 'Direction must be either "up" or "down"'

    return len(errors) == 0, errors


def validate_user_registration(data):
    """
    Validate user registration data

    Args:
        data: Request data

    Returns:
        tuple: (is_valid, errors)
    """
    errors = {}

    # Check required fields
    required_fields = ['username', 'wallet_address']
    for field in required_fields:
        if field not in data:
            errors[field] = f'Field {field} is required'

    # Validate username
    if 'username' in data:
        username = data['username']
        if not isinstance(username, str):
            errors['username'] = 'Username must be a string'
        elif len(username) < 3:
            errors['username'] = 'Username must be at least 3 characters'
        elif len(username) > 50:
            errors['username'] = 'Username cannot exceed 50 characters'

    # Validate wallet address
    if 'wallet_address' in data:
        wallet_address = data['wallet_address']
        if not isinstance(wallet_address, str):
            errors['wallet_address'] = 'Wallet address must be a string'
        elif not wallet_address.startswith('0x') and len(wallet_address) != 44:
            # Simple check for Solana addresses (base58 encoded, ~44 chars)
            errors['wallet_address'] = 'Invalid Solana wallet address'

    # Validate email if provided
    if 'email' in data and data['email']:
        email = data['email']
        if not isinstance(email, str):
            errors['email'] = 'Email must be a string'
        elif '@' not in email or '.' not in email:
            errors['email'] = 'Invalid email format'

    return len(errors) == 0, errors


def validate_login(data):
    """
    Validate login data

    Args:
        data: Request data

    Returns:
        tuple: (is_valid, errors)
    """
    errors = {}

    # Check required fields
    required_fields = ['wallet_address', 'signature', 'message']
    for field in required_fields:
        if field not in data:
            errors[field] = f'Field {field} is required'

    # Validate wallet address
    if 'wallet_address' in data:
        wallet_address = data['wallet_address']
        if not isinstance(wallet_address, str):
            errors['wallet_address'] = 'Wallet address must be a string'
        elif not wallet_address.startswith('0x') and len(wallet_address) != 44:
            # Simple check for Solana addresses (base58 encoded, ~44 chars)
            errors['wallet_address'] = 'Invalid Solana wallet address'

    # Validate signature
    if 'signature' in data:
        signature = data['signature']
        if not isinstance(signature, str):
            errors['signature'] = 'Signature must be a string'
        elif len(signature) < 10:
            errors['signature'] = 'Invalid signature'

    # Validate message
    if 'message' in data:
        message = data['message']
        if not isinstance(message, str):
            errors['message'] = 'Message must be a string'

    return len(errors) == 0, errors


def validate_nft_mint(data):
    """
    Validate NFT minting request

    Args:
        data: Request data

    Returns:
        tuple: (is_valid, errors)
    """
    errors = {}

    # Check required fields
    if 'user_id' not in data:
        errors['user_id'] = 'User ID is required'

    return len(errors) == 0, errors