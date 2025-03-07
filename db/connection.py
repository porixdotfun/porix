import pymongo
from flask import current_app, g
from werkzeug.local import LocalProxy

from utils.logging import get_logger
from config import CONFIG

logger = get_logger(__name__)


def get_db():
    """
    Get the MongoDB database connection

    Returns:
        pymongo.database.Database: MongoDB database
    """
    if 'db' not in g:
        client = get_mongo_client()
        db_name = current_app.config['MONGO_URI'].split('/')[-1]
        g.db = client[db_name]

    return g.db


def get_mongo_client():
    """
    Get the MongoDB client

    Returns:
        pymongo.MongoClient: MongoDB client
    """
    if 'mongo_client' not in g:
        mongo_uri = current_app.config['MONGO_URI']
        g.mongo_client = pymongo.MongoClient(mongo_uri)

    return g.mongo_client


def close_mongo_connection(e=None):
    """
    Close MongoDB connection

    Args:
        e: Error that caused the close (if any)
    """
    mongo_client = g.pop('mongo_client', None)

    if mongo_client is not None:
        mongo_client.close()


def init_db(app):
    """
    Initialize database connection

    Args:
        app: Flask application
    """
    # Register connection close with application teardown
    app.teardown_appcontext(close_mongo_connection)

    # Connect to database and create indexes
    with app.app_context():
        client = get_mongo_client()
        db_name = app.config['MONGO_URI'].split('/')[-1]
        db = client[db_name]

        # Create collections and indexes
        _create_collections(db)
        _create_indexes(db)

        logger.info(f"Connected to MongoDB: {app.config['MONGO_URI']}")


def _create_collections(db):
    """
    Create database collections

    Args:
        db: MongoDB database
    """
    collections = ['users', 'predictions', 'bets', 'transactions']

    for collection in collections:
        if collection not in db.list_collection_names():
            db.create_collection(collection)
            logger.info(f"Created collection: {collection}")


def _create_indexes(db):
    """
    Create database indexes

    Args:
        db: MongoDB database
    """
    # User indexes
    db.users.create_index('user_id', unique=True)
    db.users.create_index('username', unique=True)
    db.users.create_index('wallet_address', unique=True)
    logger.info("Created user indexes")

    # Prediction indexes
    db.predictions.create_index('prediction_id', unique=True)
    db.predictions.create_index('asset')
    db.predictions.create_index('prediction_type')
    db.predictions.create_index('timestamp')
    logger.info("Created prediction indexes")

    # Bet indexes
    db.bets.create_index('bet_id', unique=True)
    db.bets.create_index('user_id')
    db.bets.create_index('prediction_id')
    db.bets.create_index('status')
    db.bets.create_index('timestamp')
    logger.info("Created bet indexes")

    # Transaction indexes
    db.transactions.create_index('signature', unique=True)
    db.transactions.create_index('user_id')
    db.transactions.create_index('type')
    db.transactions.create_index('timestamp')
    logger.info("Created transaction indexes")


# Define a proxy for getting the database
db = LocalProxy(get_db)