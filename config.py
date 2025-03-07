import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    # Application settings
    APP_NAME = "Porix AI Prediction Platform"
    APP_VERSION = "1.0.0"
    DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
    TESTING = os.getenv("TESTING", "False").lower() in ("true", "1", "t")

    # API settings
    API_PREFIX = "/api/v1"
    SECRET_KEY = os.getenv("SECRET_KEY")
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
    JWT_ACCESS_TOKEN_EXPIRES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRES", "3600"))  # 1 hour

    # Database settings
    MONGO_URI = os.getenv("MONGO_URI")

    # Solana blockchain settings
    SOLANA_NETWORK = os.getenv("SOLANA_NETWORK", "devnet")
    SOLANA_RPC_URL = os.getenv("SOLANA_RPC_URL", "https://api.devnet.solana.com")

    # Token settings
    PORIX_TOKEN_ADDRESS = os.getenv("PORIX_TOKEN_ADDRESS")
    PORIX_TOKEN_DECIMALS = int(os.getenv("PORIX_TOKEN_DECIMALS", "9"))

    # AI Model settings
    MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai/models")
    MODEL_TRAINING_INTERVAL = int(os.getenv("MODEL_TRAINING_INTERVAL", "86400"))  # 24 hours in seconds

    # Prediction settings
    MIN_BET_AMOUNT = float(os.getenv("MIN_BET_AMOUNT", "1"))  # Minimum amount of PORIX tokens to place a bet
    MAX_BET_AMOUNT = float(os.getenv("MAX_BET_AMOUNT", "10000"))  # Maximum amount of PORIX tokens to place a bet
    REWARD_MULTIPLIER = float(os.getenv("REWARD_MULTIPLIER", "1.8"))  # Reward multiplier for successful predictions

    # User reputation settings
    REPUTATION_NFT_COLLECTION = os.getenv("REPUTATION_NFT_COLLECTION")

    # Logging settings
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = os.getenv("LOG_FORMAT", "{time} | {level} | {message}")

    # Third-party API keys
    COINMARKETCAP_API_KEY = os.getenv("COINMARKETCAP_API_KEY")
    TWITTER_API_KEY = os.getenv("TWITTER_API_KEY")
    TWITTER_API_SECRET = os.getenv("TWITTER_API_SECRET")

    # Feature flags
    ENABLE_MODEL_MARKETPLACE = os.getenv("ENABLE_MODEL_MARKETPLACE", "True").lower() in ("true", "1", "t")
    ENABLE_REPUTATION_SYSTEM = os.getenv("ENABLE_REPUTATION_SYSTEM", "True").lower() in ("true", "1", "t")


class DevelopmentConfig(Config):
    DEBUG = True
    LOG_LEVEL = "DEBUG"


class TestingConfig(Config):
    TESTING = True
    DEBUG = True
    LOG_LEVEL = "DEBUG"


class ProductionConfig(Config):
    DEBUG = False
    LOG_LEVEL = "WARNING"


# Set config based on environment
config_dict = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
}

# Get configuration class based on environment variable or default to development
CONFIG = config_dict[os.getenv("FLASK_ENV", "development")]