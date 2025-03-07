import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import os
from transformers import pipeline
from sklearn.decomposition import PCA

from utils.logging import get_logger
from config import CONFIG

logger = get_logger(__name__)


class DataSource(ABC):
    """Abstract base class for data sources"""

    @abstractmethod
    def fetch_data(self, asset, start_date, end_date):
        """Fetch data for the specified asset and date range"""
        pass


class CryptoDataSource(DataSource):
    """Data source for cryptocurrency price data"""

    def __init__(self, api_key=None):
        self.api_key = api_key or CONFIG.COINMARKETCAP_API_KEY

    def fetch_data(self, asset, start_date, end_date):
        """Fetch cryptocurrency historical data"""
        logger.info(f"Fetching historical data for {asset} from {start_date} to {end_date}")

        try:
            # In a real implementation, this would call an API like CoinMarketCap, CoinGecko, etc.
            # Here we'll create synthetic data for demonstration

            # Calculate number of days
            days = (end_date - start_date).days + 1

            # Generate date range
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')

            # Start with a base price
            base_price = 100.0 if asset == "PORIX" else 50000.0 if asset == "BTC" else 3000.0

            # Generate random price movements with some trend and volatility
            np.random.seed(hash(asset) % 10000)  # Use asset name as seed for consistent randomness
            daily_returns = np.random.normal(0.001, 0.02, size=len(date_range))

            # Add some trend based on the asset
            if asset == "PORIX":
                daily_returns += 0.003  # Upward trend for PORIX
            elif asset == "BTC":
                daily_returns += 0.001  # Slight upward trend for BTC

            # Calculate price series
            price_series = [base_price]
            for ret in daily_returns:
                price_series.append(price_series[-1] * (1 + ret))
            price_series = price_series[1:]  # Remove the initial base price

            # Create dataframe
            df = pd.DataFrame({
                'date': date_range,
                'open': price_series,
                'high': [p * (1 + np.random.uniform(0.01, 0.03)) for p in price_series],
                'low': [p * (1 - np.random.uniform(0.01, 0.03)) for p in price_series],
                'close': [p * (1 + np.random.normal(0, 0.01)) for p in price_series],
                'volume': [p * np.random.uniform(1000, 5000) for p in price_series],
                'market_cap': [p * np.random.uniform(10000, 50000) for p in price_series]
            })

            df = df.set_index('date')
            return df

        except Exception as e:
            logger.error(f"Error fetching data for {asset}: {str(e)}")
            return pd.DataFrame()


class NFTDataSource(DataSource):
    """Data source for NFT collection data"""

    def __init__(self, api_key=None):
        self.api_key = api_key

    def fetch_data(self, collection, start_date, end_date):
        """Fetch NFT collection historical data"""
        logger.info(f"Fetching NFT data for {collection} from {start_date} to {end_date}")

        try:
            # In a real implementation, this would call an API like OpenSea, Magic Eden, etc.
            # Here we'll create synthetic data for demonstration

            # Calculate number of days
            days = (end_date - start_date).days + 1

            # Generate date range
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')

            # Start with a base floor price
            base_floor_price = 10.0 if "rare" in collection.lower() else 1.0

            # Generate random price movements
            np.random.seed(hash(collection) % 10000)  # Use collection name as seed
            daily_changes = np.random.normal(0, 0.05, size=len(date_range))

            # Calculate floor price series
            floor_price_series = [base_floor_price]
            for change in daily_changes:
                floor_price_series.append(max(0.1, floor_price_series[-1] * (1 + change)))
            floor_price_series = floor_price_series[1:]  # Remove the initial base price

            # Create dataframe
            df = pd.DataFrame({
                'date': date_range,
                'floor_price': floor_price_series,
                'volume': [p * np.random.uniform(5, 30) for p in floor_price_series],
                'sales_count': [int(np.random.uniform(1, 100)) for _ in floor_price_series],
                'unique_buyers': [int(np.random.uniform(1, 50)) for _ in floor_price_series],
                'unique_sellers': [int(np.random.uniform(1, 30)) for _ in floor_price_series],
                'listings_count': [int(np.random.uniform(10, 200)) for _ in floor_price_series]
            })

            df = df.set_index('date')
            return df

        except Exception as e:
            logger.error(f"Error fetching NFT data for {collection}: {str(e)}")
            return pd.DataFrame()


class SocialSentimentAnalyzer:
    """Analyzes social media sentiment for assets"""

    def __init__(self):
        # Initialize sentiment analysis model
        try:
            self.sentiment_pipeline = pipeline("sentiment-analysis")
            self.sentiment_model_loaded = True
        except Exception as e:
            logger.error(f"Error loading sentiment model: {str(e)}")
            self.sentiment_model_loaded = False

    def analyze_sentiment(self, asset, days=7):
        """
        Analyze social media sentiment for the given asset

        Args:
            asset: Asset to analyze
            days: Number of days of data to analyze

        Returns:
            dict: Sentiment analysis results
        """
        # In a real implementation, this would fetch social media data
        # Here we'll simulate sentiment scores

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        # Generate synthetic sentiment data
        np.random.seed(hash(asset) % 10000)

        # Raw social media posts (simulated)
        example_texts = [
            f"Just bought some more {asset}! To the moon! 🚀",
            f"Not sure about {asset}, seem overvalued right now",
            f"{asset} is the future of finance, holding strong!",
            f"Disappointed with the latest {asset} announcement",
            f"Massive growth potential for {asset}, bullish!"
        ]

        # Analyze sentiment if model is loaded
        sentiments = []
        if self.sentiment_model_loaded:
            for text in example_texts:
                try:
                    result = self.sentiment_pipeline(text)[0]
                    sentiments.append({
                        "text": text,
                        "label": result["label"],
                        "score": result["score"]
                    })
                except Exception as e:
                    logger.error(f"Error analyzing sentiment: {str(e)}")

        # Create daily sentiment scores (simulated)
        daily_sentiment = []
        for date in date_range:
            sentiment_score = np.random.normal(0.6, 0.2)  # Slightly positive bias
            volume = int(np.random.uniform(100, 1000))
            daily_sentiment.append({
                "date": date.strftime("%Y-%m-%d"),
                "sentiment_score": max(0, min(1, sentiment_score)),  # Clamp between 0 and 1
                "volume": volume,
                "bullish_percentage": max(0, min(100, sentiment_score * 100)),
                "bearish_percentage": max(0, min(100, (1 - sentiment_score) * 100))
            })

        # Overall sentiment
        overall_sentiment = np.mean([day["sentiment_score"] for day in daily_sentiment])

        return {
            "asset": asset,
            "overall_sentiment": overall_sentiment,
            "sentiment_label": "bullish" if overall_sentiment > 0.6 else "neutral" if overall_sentiment > 0.4 else "bearish",
            "daily_sentiment": daily_sentiment,
            "example_analyses": sentiments
        }


class DataProcessor:
    """Processes data for AI models"""

    def __init__(self):
        self.crypto_source = CryptoDataSource()
        self.nft_source = NFTDataSource()
        self.sentiment_analyzer = SocialSentimentAnalyzer()

    def fetch_historical_data(self, asset, start_date, end_date):
        """
        Fetch historical data for the given asset

        Args:
            asset: Asset identifier (crypto symbol, NFT collection, etc.)
            start_date: Start date for data
            end_date: End date for data

        Returns:
            pd.DataFrame: Historical data
        """
        # Determine data source based on asset type
        if asset.startswith("NFT:"):
            # NFT collection
            collection = asset[4:]  # Remove "NFT:" prefix
            return self.nft_source.fetch_data(collection, start_date, end_date)
        else:
            # Assume cryptocurrency
            return self.crypto_source.fetch_data(asset, start_date, end_date)

    def create_price_features(self, data):
        """
        Create features for price prediction

        Args:
            data: DataFrame with price data

        Returns:
            pd.DataFrame: DataFrame with additional features
        """
        # Make a copy to avoid modifying the original
        features = data.copy()

        # Basic technical indicators

        # Moving averages
        features['ma7'] = features['close'].rolling(window=7).mean()
        features['ma14'] = features['close'].rolling(window=14).mean()
        features['ma30'] = features['close'].rolling(window=30).mean()

        # Price momentum
        features['momentum_1d'] = features['close'].pct_change(periods=1)
        features['momentum_3d'] = features['close'].pct_change(periods=3)
        features['momentum_7d'] = features['close'].pct_change(periods=7)

        # Volatility
        features['volatility_7d'] = features['close'].rolling(window=7).std() / features['close'].rolling(
            window=7).mean()

        # Volume indicators
        if 'volume' in features.columns:
            features['volume_change_1d'] = features['volume'].pct_change(periods=1)
            features['volume_ma7'] = features['volume'].rolling(window=7).mean()
            features['volume_ma14'] = features['volume'].rolling(window=14).mean()

            # Price-volume relationship
            features['price_volume_ratio'] = features['close'] / features['volume']
            features['price_volume_ratio_ma7'] = features['price_volume_ratio'].rolling(window=7).mean()

        # Return features with NaN values from calculations
        return features

    def create_nft_features(self, data):
        """
        Create features for NFT floor price prediction

        Args:
            data: DataFrame with NFT data

        Returns:
            pd.DataFrame: DataFrame with additional features
        """
        # Make a copy to avoid modifying the original
        features = data.copy()

        # Floor price moving averages
        features['floor_ma3'] = features['floor_price'].rolling(window=3).mean()
        features['floor_ma7'] = features['floor_price'].rolling(window=7).mean()
        features['floor_ma14'] = features['floor_price'].rolling(window=14).mean()

        # Floor price momentum
        features['floor_momentum_1d'] = features['floor_price'].pct_change(periods=1)
        features['floor_momentum_3d'] = features['floor_price'].pct_change(periods=3)

        # Volume indicators
        if 'volume' in features.columns:
            features['volume_change_1d'] = features['volume'].pct_change(periods=1)
            features['volume_ma7'] = features['volume'].rolling(window=7).mean()

        # Sales indicators
        if 'sales_count' in features.columns:
            features['sales_change_1d'] = features['sales_count'].pct_change(periods=1)
            features['sales_ma7'] = features['sales_count'].rolling(window=7).mean()

        # Liquidity indicators
        if 'listings_count' in features.columns and 'sales_count' in features.columns:
            features['liquidity_ratio'] = features['sales_count'] / features['listings_count']
            features['liquidity_ma7'] = features['liquidity_ratio'].rolling(window=7).mean()

        return features

    def enrich_data_with_sentiment(self, data, asset):
        """
        Enrich data with sentiment analysis

        Args:
            data: DataFrame with price/NFT data
            asset: Asset identifier

        Returns:
            pd.DataFrame: Enriched data
        """
        # Analyze sentiment
        sentiment_data = self.sentiment_analyzer.analyze_sentiment(asset)

        # Create sentiment dataframe
        sentiment_df = pd.DataFrame(sentiment_data['daily_sentiment'])
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
        sentiment_df = sentiment_df.set_index('date')

        # Merge with original data
        merged_data = data.merge(sentiment_df, left_index=True, right_index=True, how='left')

        # Fill missing sentiment data
        merged_data['sentiment_score'] = merged_data['sentiment_score'].fillna(0.5)
        merged_data['bullish_percentage'] = merged_data['bullish_percentage'].fillna(50)
        merged_data['bearish_percentage'] = merged_data['bearish_percentage'].fillna(50)

        return merged_data

    def apply_dimension_reduction(self, features, n_components=10):
        """
        Apply dimension reduction to features

        Args:
            features: Feature matrix (numpy array)
            n_components: Number of components to keep

        Returns:
            numpy.ndarray: Reduced features
        """
        if features.shape[1] <= n_components:
            return features

        pca = PCA(n_components=min(n_components, features.shape[1]))
        return pca.fit_transform(features)