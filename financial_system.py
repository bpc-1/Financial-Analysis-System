# financial_system.py - Main system class

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

from data_fetcher import DataFetcher
from technical_analysis import TechnicalAnalyzer
from signal_generator import SignalGenerator
from portfolio_optimizer import PortfolioOptimizer
from backtester import Backtester
from ml_predictor import MLPredictor
from visualizer import Visualizer
from config import CONFIG

logger = logging.getLogger(__name__)

class FinancialAnalysisSystem:
    """Main class for the financial analysis system"""
    
    def __init__(self, config=CONFIG):
        """Initialize the system with its components"""
        self.config = config
        self.data_fetcher = DataFetcher(config)
        self.technical_analyzer = TechnicalAnalyzer()
        self.signal_generator = SignalGenerator()
        self.portfolio_optimizer = PortfolioOptimizer()
        self.backtester = Backtester()
        self.ml_predictor = MLPredictor()
        self.visualizer = Visualizer()
    
    def update_data(self):
        """Update all configured data sources"""
        self.data_fetcher.update_all_data()
    
    def analyze_asset(self, asset_type, ticker, start_date=None, end_date=None, period=None, include_sentiment=True):
        """Analyze a specific asset and generate signals"""
        logger.info(f"Analyzing {asset_type}: {ticker}")
        
        # Fetch data
        if asset_type == 'stock':
            if period:
                stock_data = self.data_fetcher.fetch_stock_data([ticker], period)
                if ticker not in stock_data:
                    raise ValueError(f"Failed to fetch data for {ticker}")
                data = stock_data[ticker]
            else:
                data = self.data_fetcher.get_stock_data_from_db(ticker, start_date, end_date)
        elif asset_type == 'crypto':
            if period:
                days = 365 if period == '1y' else int(period[:-1]) * 30 if period.endswith('mo') else int(period[:-1])
                crypto_data = self.data_fetcher.fetch_crypto_data([ticker], days)
                if ticker not in crypto_data:
                    raise ValueError(f"Failed to fetch data for {ticker}")
                data = crypto_data[ticker]
                data = data.rename(columns={'price': 'close'})
            else:
                data = self.data_fetcher.get_crypto_data_from_db(ticker, start_date, end_date)
                data = data.rename(columns={'price': 'close'})
        else:
            raise ValueError(f"Unsupported asset type: {asset_type}")
        
        # Get sentiment data if requested
        sentiment_data = None
        if include_sentiment:
            try:
                sentiment_data = self.data_fetcher.get_sentiment_data_from_db(ticker)
            except:
                # Generate mock sentiment data if not available
                sentiment_data = self.data_fetcher.fetch_sentiment_data([ticker])[ticker]
        
        # Perform technical analysis
        analysis_result = self.technical_analyzer.analyze_ticker(data)
        
        # Combine with sentiment data if available
        if sentiment_data is not None:
            analysis_result = self.technical_analyzer.combine_with_sentiment(analysis_result, sentiment_data)
        
        # Generate signals
        with_signals = self.signal_generator.generate_combined_signal(analysis_result)
        
        # Backtest strategy
        backtest_result, metrics = self.backtester.backtest_strategy(with_signals)
        
        logger.info(f"Analysis completed for {ticker}, metrics: {metrics}")
        
        return backtest_result, metrics
    
    def get_portfolio_recommendations(self, asset_type='stock', top_n=5, period='3mo'):
        """Get portfolio recommendations based on analysis of multiple assets"""
        if asset_type == 'stock':
            tickers = self.config['stock_tickers']
        elif asset_type == 'crypto':
            tickers = self.config['crypto_tickers']
        else:
            raise ValueError(f"Unsupported asset type: {asset_type}")
        
        results = []
        analysis_data = {}
        
        for ticker in tickers:
            try:
                backtest_result, metrics = self.analyze_asset(asset_type, ticker, period=period)
                analysis_data[ticker] = backtest_result
                
                results.append({
                    'ticker': ticker,
                    'sharpe_ratio': metrics['sharpe_ratio'],
                    'annualized_return': metrics['annualized_return'],
                    'max_drawdown': metrics['max_drawdown'],
                    'signal': backtest_result['final_signal'].iloc[-1]
                })
            except Exception as e:
                logger.error(f"Error analyzing {ticker}: {e}")
        
        # Sort by Sharpe ratio
        sorted_results = sorted(results, key=lambda x: x['sharpe_ratio'], reverse=True)
        
        # Get top N tickers
        top_tickers = [result['ticker'] for result in sorted_results[:top_n]]
        
        # Extract close prices for portfolio optimization
        close_prices = {}
        for ticker in top_tickers:
            if ticker in analysis_data:
                close_prices[ticker] = analysis_data[ticker]['close']
        
        # Calculate returns
        returns = pd.DataFrame(close_prices).pct_change().dropna()
        
        # Optimize portfolio weights
        optimized_portfolio = self.portfolio_optimizer.optimize_weights(returns, method='max_sharpe')
        
        # Generate efficient frontier
        efficient_frontier = self.portfolio_optimizer.generate_efficient_frontier(returns)
        
        return {
            'top_assets': sorted_results[:top_n],
            'optimized_weights': optimized_portfolio['weights'],
            'performance': optimized_portfolio['performance'],
            'efficient_frontier': efficient_frontier
        }
    
    def visualize_analysis(self, backtest_result, metrics, title=None):
        """Visualize the analysis results"""
        price_chart = self.visualizer.plot_price_with_indicators(backtest_result, title=f"{title} - Technical Analysis")
        backtest_chart = self.visualizer.plot_backtest_results(backtest_result, metrics, title=f"{title} - Backtest Results")
        
        return price_chart, backtest_chart
    
    def visualize_portfolio(self, portfolio_recommendations):
        """Visualize the portfolio recommendations"""
        # Plot portfolio allocation
        allocation_chart = self.visualizer.plot_portfolio_allocation(
            portfolio_recommendations['optimized_weights'],
            title="Optimized Portfolio Allocation"
        )
        
        # Plot efficient frontier
        frontier_chart = self.visualizer.plot_efficient_frontier(
            portfolio_recommendations['efficient_frontier'],
            figsize=(10, 6)
        )
        
        return allocation_chart, frontier_chart
    
    def train_ml_model(self, asset_type, ticker, period='1y', prediction_horizon=5, prediction_type='direction'):
        """Train a machine learning model for price prediction"""
        # Fetch and analyze data
        backtest_result, _ = self.analyze_asset(asset_type, ticker, period=period, include_sentiment=True)
        
        # Train model based on prediction type
        if prediction_type == 'direction':
            ml_results = self.ml_predictor.train_direction_model(
                backtest_result, 
                prediction_horizon=prediction_horizon
            )
        else:
            ml_results = self.ml_predictor.train_price_model(
                backtest_result,
                target_column=prediction_type,
                prediction_horizon=prediction_horizon
            )
        
        # Perform cross-validation
        cv_results = self.ml_predictor.crossvalidate_model(
            backtest_result,
            target_column=prediction_type,
            prediction_horizon=prediction_horizon
        )
        
        # Generate predictions
        predictions = self.ml_predictor.predict(backtest_result)
        
        # Add predictions to backtest results
        prediction_results = backtest_result.copy()
        prediction_results['prediction'] = pd.Series(predictions)
        
        # Generate signals based on predictions
        prediction_results = self.ml_predictor.generate_trading_signals(prediction_results)
        
        # Backtest ML strategy
        ml_backtest, ml_metrics = self.backtester.backtest_strategy(prediction_results)
        
        return {
            'training_results': ml_results,
            'cross_validation': cv_results,
            'backtest_results': ml_backtest,
            'backtest_metrics': ml_metrics,
            'feature_importance': self.ml_predictor.feature_importance
        }
    
    def analyze_portfolio_risk(self, portfolio_symbols, weights=None, period='1y'):
        """Analyze portfolio risk metrics"""
        # Fetch data for all symbols
        price_data = {}
        for symbol in portfolio_symbols:
            try:
                # Determine asset type (stock or crypto)
                if symbol in self.config['stock_tickers']:
                    asset_type = 'stock'
                elif symbol in self.config['crypto_tickers']:
                    asset_type = 'crypto'
                else:
                    # Default to stock
                    asset_type = 'stock'
                
                # Fetch data
                df, _ = self.analyze_asset(asset_type, symbol, period=period, include_sentiment=False)
                price_data[symbol] = df['close']
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
        
        # Calculate returns
        returns = pd.DataFrame(price_data).pct_change().dropna()
        
        # Use equal weights if not provided
        if weights is None:
            weights = {symbol: 1/len(portfolio_symbols) for symbol in portfolio_symbols}
        
        # Calculate portfolio returns
        portfolio_returns = returns.dot(list(weights.values()))
        
        # Calculate risk metrics
        risk_metrics = self.portfolio_optimizer.calculate_risk_metrics(returns)
        
        # Calculate portfolio risk metrics
        portfolio_risk = self.portfolio_optimizer.calculate_risk_metrics(pd.DataFrame(portfolio_returns))
        
        # Run Monte Carlo simulation
        simulation_results = self.backtester.monte_carlo_simulation(
            pd.DataFrame({'close': (1 + portfolio_returns).cumprod(), 'returns': portfolio_returns})
        )
        
        # Calculate correlation matrix
        correlation_matrix = returns.corr()
        
        return {
            'asset_metrics': risk_metrics,
            'portfolio_metrics': portfolio_risk,
            'correlation_matrix': correlation_matrix,
            'simulation_results': simulation_results
        }
    
    def backtest_portfolio_strategy(self, portfolio_symbols, weights=None, period='1y', rebalance_frequency='M'):
        """Backtest a portfolio strategy with periodic rebalancing"""
        # Fetch data for all symbols
        asset_data = {}
        for symbol in portfolio_symbols:
            try:
                # Determine asset type (stock or crypto)
                if symbol in self.config['stock_tickers']:
                    asset_type = 'stock'
                elif symbol in self.config['crypto_tickers']:
                    asset_type = 'crypto'
                else:
                    # Default to stock
                    asset_type = 'stock'
                
                # Fetch data
                df, _ = self.analyze_asset(asset_type, symbol, period=period, include_sentiment=False)
                asset_data[symbol] = df
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
        
        # Use equal weights if not provided
        if weights is None:
            weights = {symbol: 1/len(portfolio_symbols) for symbol in portfolio_symbols}
        
        # Backtest portfolio
        portfolio_results, portfolio_metrics = self.backtester.backtest_portfolio(
            asset_data, weights, rebalance_frequency=rebalance_frequency
        )
        
        return portfolio_results, portfolio_metrics