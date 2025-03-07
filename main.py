import os
from flask import Flask
from flask_cors import CORS

from config import CONFIG
from api.routes import register_routes
from db.connection import init_db
from utils.logging import setup_logging


def create_app(config=CONFIG):
    """
    Factory function to create and configure the Flask application
    """
    # Initialize Flask application
    app = Flask(__name__)
    app.config.from_object(config)

    # Setup CORS
    CORS(app)

    # Setup logging
    setup_logging(app)

    # Initialize database connection
    init_db(app)

    # Register API routes
    register_routes(app)

    app.logger.info(f"Application {app.config['APP_NAME']} v{app.config['APP_VERSION']} started")
    app.logger.info(f"Running in {os.getenv('FLASK_ENV', 'development')} mode")

    return app


if __name__ == "__main__":
    app = create_app()

    # Run the application
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 5000))
    debug = app.config["DEBUG"]

    app.run(host=host, port=port, debug=debug)