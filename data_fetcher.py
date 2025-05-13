# data_fetcher.py - For data gathering functionality

import pandas as pd
import numpy as np
import sqlite3
import requests
import yfinance as yf
import logging
from datetime import datetime, timedelta
import re
import time

from utils import normalize_dataframe
from config import CONFIG

logger = logging.getLogger(__name__)

class DataFetcher:
    """Class for fetching financial data from various sources"""
    
    def __init__(self, config=CONFIG):
        """Initialize with configuration"""
        self.config = config
        self.db_conn = self._init_database()
        
    def _init_database(self):
        """Initialize database connection and tables"""
        conn = sqlite3.connect(self.config["database_path"])
        
        # Create tables if they don't exist
        cursor = conn.cursor()
        
        # Stock price table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_prices (
            ticker TEXT,
            date TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            PRIMARY KEY (ticker, date)
        )
        ''')
        
        # Crypto price table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS crypto_prices (
            ticker TEXT,
            date TEXT,
            price REAL,
            market_cap REAL,
            volume REAL,
            PRIMARY KEY (ticker, date)
        )
        ''')
        
        # Economic indicators table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS economic_indicators (
            indicator TEXT,
            date TEXT,
            value REAL,
            PRIMARY KEY (indicator, date)
        )
        ''')
        
        # Sentiment data table (new)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS sentiment_data (
            ticker TEXT,
            date TEXT,
            sentiment_score REAL,
            volume INTEGER,
            PRIMARY KEY (ticker, date)
        )
        ''')
        
        conn.commit()
        return conn
    
    def fetch_stock_data(self, tickers=None, period="1y"):
        """Fetch stock data for the given tickers"""
        tickers = tickers or self.config["stock_tickers"]
        logger.info(f"Fetching stock data for {tickers}")
        
        all_data = {}
        for ticker in tickers:
            try:
                # Download data
                ticker_data = yf.download(ticker, period=period)
                
                # Debug info to see what's coming from yfinance
                logger.info(f"Downloaded data for {ticker}, shape: {ticker_data.shape}")
                logger.info(f"Column types: {type(ticker_data.columns)}")
                logger.info(f"Original columns: {ticker_data.columns.tolist() if not isinstance(ticker_data.columns, pd.MultiIndex) else list(ticker_data.columns)}")
                
                # Normalize column names - handle both multi-index and regular columns
                ticker_data = normalize_dataframe(ticker_data, ticker)
                
                # Store in our results dictionary
                all_data[ticker] = ticker_data
                
                # Debug info after normalization
                logger.info(f"Normalized columns: {ticker_data.columns.tolist()}")
                
                # Save to database
                df = ticker_data.reset_index()
                
                # Make sure required columns exist
                required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
                if not all(col in df.columns for col in required_cols):
                    logger.warning(f"Missing required columns for {ticker}: {set(required_cols) - set(df.columns)}")
                    logger.warning(f"Available columns: {df.columns.tolist()}")
                    continue
                
                for _, row in df.iterrows():
                    self.db_conn.execute('''
                    INSERT OR REPLACE INTO stock_prices 
                    (ticker, date, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        ticker, 
                        row['date'].strftime('%Y-%m-%d'), 
                        row['open'], 
                        row['high'], 
                        row['low'], 
                        row['close'], 
                        row['volume']
                    ))
                
                self.db_conn.commit()
                logger.info(f"Successfully fetched and stored data for {ticker}")
                
            except Exception as e:
                logger.error(f"Error fetching data for {ticker}: {e}")
                logger.exception("Full stack trace:")
        
        return all_data
    
    def fetch_crypto_data(self, tickers=None, days=365):
        """Fetch cryptocurrency data for the given tickers"""
        tickers = tickers or self.config["crypto_tickers"]
        logger.info(f"Fetching crypto data for {tickers}")
        
        all_data = {}
        for ticker in tickers:
            try:
                url = f"{self.config['crypto_api_url']}/coins/{ticker}/market_chart"
                params = {
                    'vs_currency': 'usd',
                    'days': days,
                    'interval': 'daily'
                }
                
                response = requests.get(url, params=params)
                data = response.json()
                
                # Process price data
                prices = data['prices']
                market_caps = data['market_caps']
                volumes = data['total_volumes']
                
                df = pd.DataFrame(prices, columns=['timestamp', 'price'])
                df['market_cap'] = [mc[1] for mc in market_caps]
                df['volume'] = [vol[1] for vol in volumes]
                
                # Convert timestamp to date
                df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.drop('timestamp', axis=1)
                
                all_data[ticker] = df
                
                # Save to database
                for _, row in df.iterrows():
                    self.db_conn.execute('''
                    INSERT OR REPLACE INTO crypto_prices 
                    (ticker, date, price, market_cap, volume)
                    VALUES (?, ?, ?, ?, ?)
                    ''', (
                        ticker, 
                        row['date'].strftime('%Y-%m-%d'), 
                        row['price'], 
                        row['market_cap'], 
                        row['volume']
                    ))
                
                self.db_conn.commit()
                logger.info(f"Successfully fetched and stored data for {ticker}")
                
            except Exception as e:
                logger.error(f"Error fetching data for {ticker}: {e}")
        
        return all_data
    
    def fetch_economic_data(self, indicators=None):
        """Fetch economic indicators from FRED (would require fredapi package)"""
        indicators = indicators or self.config["economic_indicators"]
        logger.info(f"Fetching economic indicators: {indicators}")
        
        # This is a placeholder - in a real implementation, you would use the FRED API
        # import fredapi as fa
        # fred = fa.Fred(api_key='your_api_key_here')
        
        # For now, we'll just create mock data
        all_data = {}
        for indicator in indicators:
            try:
                # Create mock data
                start_date = datetime.now() - timedelta(days=365*2)  # 2 years of data
                dates = pd.date_range(start=start_date, end=datetime.now(), freq='M')
                
                if indicator == "GDP":
                    # Quarterly growing values
                    values = np.linspace(21000, 23000, len(dates)) + np.random.normal(0, 100, len(dates))
                elif indicator == "UNRATE":
                    # Unemployment rate - fluctuating around 3-5%
                    values = 4 + np.sin(np.linspace(0, 4*np.pi, len(dates))) + np.random.normal(0, 0.2, len(dates))
                elif indicator == "CPIAUCSL":
                    # CPI - gradually increasing
                    values = np.linspace(280, 300, len(dates)) + np.random.normal(0, 1, len(dates))
                else:
                    # Generic economic indicator
                    values = 100 + np.cumsum(np.random.normal(0.1, 0.5, len(dates)))
                
                df = pd.DataFrame({
                    'date': dates,
                    'value': values
                })
                
                all_data[indicator] = df
                
                # Save to database
                for _, row in df.iterrows():
                    self.db_conn.execute('''
                    INSERT OR REPLACE INTO economic_indicators 
                    (indicator, date, value)
                    VALUES (?, ?, ?)
                    ''', (
                        indicator, 
                        row['date'].strftime('%Y-%m-%d'), 
                        row['value']
                    ))
                
                self.db_conn.commit()
                logger.info(f"Successfully created mock data for economic indicator: {indicator}")
                
            except Exception as e:
                logger.error(f"Error creating mock data for economic indicator {indicator}: {e}")
        
        return all_data
    
    def fetch_sentiment_data(self, tickers=None, days=30):
        """Fetch or generate sentiment data for the given tickers"""
        tickers = tickers or self.config["stock_tickers"]
        logger.info(f"Generating sentiment data for {tickers}")
        
        # This is a placeholder - in a real implementation, you would use a news sentiment API
        # For now, we'll just create mock data
        all_data = {}
        for ticker in tickers:
            try:
                # Create mock data
                dates = pd.date_range(end=datetime.now(), periods=days)
                
                # Create sentiment scores with some randomness and trend
                # Companies tend to have positive sentiment on average
                base_sentiment = 0.2  
                sentiment = base_sentiment + np.random.normal(0, 0.3, days)
                
                # Add some trending behavior
                trend = np.sin(np.linspace(0, 3*np.pi, days)) * 0.2
                sentiment += trend
                
                # Clip to reasonable range
                sentiment = np.clip(sentiment, -1, 1)
                
                # News volume (activity)
                volume = np.random.randint(10, 100, size=days)
                
                df = pd.DataFrame({
                    'date': dates,
                    'sentiment_score': sentiment,
                    'volume': volume
                })
                
                all_data[ticker] = df
                
                # Save to database
                for _, row in df.iterrows():
                    self.db_conn.execute('''
                    INSERT OR REPLACE INTO sentiment_data 
                    (ticker, date, sentiment_score, volume)
                    VALUES (?, ?, ?, ?)
                    ''', (
                        ticker, 
                        row['date'].strftime('%Y-%m-%d'), 
                        row['sentiment_score'], 
                        row['volume']
                    ))
                
                self.db_conn.commit()
                logger.info(f"Successfully generated sentiment data for {ticker}")
                
            except Exception as e:
                logger.error(f"Error generating sentiment data for {ticker}: {e}")
        
        return all_data
    
    def get_stock_data_from_db(self, ticker, start_date=None, end_date=None):
        """Retrieve stock data from the database"""
        query = "SELECT * FROM stock_prices WHERE ticker = ?"
        params = [ticker]
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
            
        query += " ORDER BY date"
        
        df = pd.read_sql_query(query, self.db_conn, params=params)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        return df
    
    def get_crypto_data_from_db(self, ticker, start_date=None, end_date=None):
        """Retrieve cryptocurrency data from the database"""
        query = "SELECT * FROM crypto_prices WHERE ticker = ?"
        params = [ticker]
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
            
        query += " ORDER BY date"
        
        df = pd.read_sql_query(query, self.db_conn, params=params)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        return df
    
    def get_economic_data_from_db(self, indicator, start_date=None, end_date=None):
        """Retrieve economic data from the database"""
        query = "SELECT * FROM economic_indicators WHERE indicator = ?"
        params = [indicator]
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
            
        query += " ORDER BY date"
        
        df = pd.read_sql_query(query, self.db_conn, params=params)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        return df
    
    def get_sentiment_data_from_db(self, ticker, start_date=None, end_date=None):
        """Retrieve sentiment data from the database"""
        query = "SELECT * FROM sentiment_data WHERE ticker = ?"
        params = [ticker]
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
            
        query += " ORDER BY date"
        
        df = pd.read_sql_query(query, self.db_conn, params=params)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        return df

    def update_all_data(self):
        """Update all configured data sources"""
        self.fetch_stock_data()
        self.fetch_crypto_data()
        self.fetch_economic_data()
        self.fetch_sentiment_data()