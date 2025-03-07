![Porix Logo](/porix.png)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Solana](https://img.shields.io/badge/Solana-Compatible-9945FF.svg)](https://solana.com/)
[![MongoDB](https://img.shields.io/badge/MongoDB-4EA94B?logo=mongodb&logoColor=white)](https://www.mongodb.com/)
[![Flask](https://img.shields.io/badge/Flask-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![X](https://img.shields.io/badge/X-000000?logo=x&logoColor=white)](https://twitter.com/porixdotfun)
[![Website](https://img.shields.io/badge/Website-FF7139?logo=firefox-browser&logoColor=white)](https://porix.fun)
[![GitHub](https://img.shields.io/badge/GitHub-181717?logo=github&logoColor=white)](https://github.com/porixdotfun)

Porix is an AI-powered prediction and betting platform on Solana blockchain, allowing users to bet on cryptocurrency prices, NFT floor prices, and other market metrics.

## Overview

Porix uses advanced AI models to analyze and predict various crypto market indicators, including:

- Cryptocurrency price movements (up/down)
- Specific price targets
- NFT collection floor prices
- Trading volume changes
- Project event occurrences

Users can bet PORIX tokens on these predictions and receive rewards for correct predictions. The AI models analyze historical data, news sentiment, on-chain analytics, and social media trends to make comprehensive predictions.

## Key Features

### AI-Powered Predictions
- Multiple prediction types (price direction, price targets, NFT floors)
- Varying time horizons (24h, 7d, etc.)
- Confidence scores with each prediction

### Betting Mechanism
- Bet PORIX tokens on predictions
- Higher rewards for less likely outcomes
- Automatic settlement when predictions mature

### Reputation System
- Mint NFTs representing prediction history
- User rankings based on prediction accuracy
- Leaderboards for top predictors

### Model Marketplace
- Advanced users can train and share AI models
- Model creators earn revenue when their models are used
- Community voting on model improvements

## Technology Stack

- **Backend**: Python with Flask for the API server
- **Blockchain**: Solana for token transactions and NFTs
- **AI**: TensorFlow and scikit-learn for prediction models
- **Database**: MongoDB for data persistence
- **Authentication**: Solana wallet signatures

## Installation

### Prerequisites

- Python 3.9+
- MongoDB
- Solana CLI tools

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/porixdotfun/porix.git
   cd porix
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your configuration:
   ```
   FLASK_ENV=development
   DEBUG=True
   MONGO_URI=mongodb://localhost:27017/porix
   SOLANA_NETWORK=devnet
   SOLANA_RPC_URL=https://api.devnet.solana.com
   SECRET_KEY=your-secret-key
   JWT_SECRET_KEY=your-jwt-secret-key
   ```

5. Run the application:
   ```
   python main.py
   ```

## API Documentation

### Authentication

All API endpoints except for registration and login require authentication using a JWT token.

```
Authorization: Bearer <token>
```

### Endpoints

#### User Management
- `POST /api/v1/user/register` - Register a new user
- `POST /api/v1/user/login` - Log in with wallet signature
- `GET /api/v1/user/<user_id>` - Get user details
- `GET /api/v1/user/<user_id>/balance` - Get user token balance
- `GET /api/v1/user/<user_id>/stats` - Get user statistics

#### Predictions
- `POST /api/v1/prediction/price-direction` - Get price direction prediction
- `POST /api/v1/prediction/price-target` - Get price target prediction
- `POST /api/v1/prediction/nft-floor` - Get NFT floor price prediction
- `GET /api/v1/prediction/history` - Get prediction history
- `GET /api/v1/prediction/<prediction_id>` - Get prediction details
- `GET /api/v1/prediction/<prediction_id>/outcome` - Get prediction outcome

#### Betting
- `POST /api/v1/betting/place` - Place a bet
- `GET /api/v1/betting/history` - Get betting history
- `GET /api/v1/betting/<bet_id>` - Get bet details
- `POST /api/v1/betting/<bet_id>/cancel` - Cancel a bet
- `GET /api/v1/betting/leaderboard` - Get betting leaderboard

#### Reputation
- `POST /api/v1/reputation/mint-nft` - Mint reputation NFT
- `GET /api/v1/reputation/<user_id>/nfts` - Get user's NFTs
- `GET /api/v1/reputation/leaderboard` - Get reputation leaderboard

#### Platform
- `GET /api/v1/health` - Health check
- `GET /api/v1/stats` - Platform statistics

## License

MIT License

## Contact

For more information, contact the Porix team at team@porix.fun or join our Discord community.