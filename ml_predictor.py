# ml_predictor.py - Machine learning for predictions

import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score

logger = logging.getLogger(__name__)

class MLPredictor:
    """Class for machine learning predictions"""
    
    def __init__(self):
        """Initialize ML predictor"""
        self.model = None
        self.feature_importance = None
        self.scaler = StandardScaler()
    
    def _prepare_features(self, df, target_column, prediction_horizon=5, train_size=0.8):
        """
        Prepare features and target variables for machine learning
        
        Args:
            df: DataFrame with technical indicators
            target_column: Column to predict ('close', 'returns', etc.)
            prediction_horizon: How many days into the future to predict
            train_size: Fraction of data to use for training
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Create a copy of the dataframe
        data = df.copy()
        
        # Create target variable (future price/return)
        if target_column == 'direction':
            # Binary classification: price direction (up/down)
            data['target'] = np.where(data['close'].shift(-prediction_horizon) > data['close'], 1, 0)
        elif target_column == 'returns':
            # Regression: future returns
            data['target'] = data['close'].pct_change(prediction_horizon).shift(-prediction_horizon)
        else:
            # Regression: future prices
            data['target'] = data[target_column].shift(-prediction_horizon)
        
        # Drop rows with NaN in target
        data = data.dropna(subset=['target'])
        
        # Define features to use (all technical indicators)
        # Exclude non-feature columns
        exclude_columns = ['target', 'open', 'high', 'low', 'close', 'volume',
                          'returns', 'cumulative_returns', 'drawdown',
                          'position', 'cash', 'holdings', 'portfolio_value',
                          'trade_type', 'trade_price', 'trade_shares']
        
        # Filter columns to include only numeric ones
        feature_columns = [col for col in data.columns 
                          if col not in exclude_columns 
                          and np.issubdtype(data[col].dtype, np.number)]
        
        # Handle NaN values in features
        data = data.dropna(subset=feature_columns)
        
        # Split data chronologically
        train_idx = int(len(data) * train_size)
        
        X_train = data[feature_columns][:train_idx]
        X_test = data[feature_columns][train_idx:]
        y_train = data['target'][:train_idx]
        y_test = data['target'][train_idx:]
        
        # Standardize features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test, feature_columns
    
    def train_direction_model(self, df, prediction_horizon=5, train_size=0.8):
        """
        Train a classification model to predict price direction
        
        Args:
            df: DataFrame with technical indicators
            prediction_horizon: How many days into the future to predict
            train_size: Fraction of data to use for training
        
        Returns:
            Dictionary with model performance metrics
        """
        X_train, X_test, y_train, y_test, feature_columns = self._prepare_features(
            df, 'direction', prediction_horizon, train_size
        )
        
        # Train Random Forest classifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Calculate feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': model.feature_importances_
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        self.model = model
        self.feature_importance = feature_importance
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'feature_importance': feature_importance
        }
    
    def train_price_model(self, df, target_column='close', prediction_horizon=5, train_size=0.8):
        """
        Train a regression model to predict future prices or returns
        
        Args:
            df: DataFrame with technical indicators
            target_column: Column to predict ('close', 'returns')
            prediction_horizon: How many days into the future to predict
            train_size: Fraction of data to use for training
        
        Returns:
            Dictionary with model performance metrics
        """
        X_train, X_test, y_train, y_test, feature_columns = self._prepare_features(
            df, target_column, prediction_horizon, train_size
        )
        
        # Train Random Forest regressor
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': model.feature_importances_
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        self.model = model
        self.feature_importance = feature_importance
        
        return {
            'mse': mse,
            'rmse': rmse,
            'r2_score': r2,
            'feature_importance': feature_importance
        }
    
    def predict(self, df):
        """
        Make predictions using the trained model
        
        Args:
            df: DataFrame with technical indicators
        
        Returns:
            Series with predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_direction_model or train_price_model first.")
        
        # Define features to use (same as in training)
        exclude_columns = ['target', 'open', 'high', 'low', 'close', 'volume',
                          'returns', 'cumulative_returns', 'drawdown',
                          'position', 'cash', 'holdings', 'portfolio_value',
                          'trade_type', 'trade_price', 'trade_shares']
        
        # Filter columns to include only numeric ones
        feature_columns = [col for col in df.columns 
                          if col not in exclude_columns 
                          and np.issubdtype(df[col].dtype, np.number)]
        
        # Handle NaN values in features
        data = df.dropna(subset=feature_columns)
        
        # Standardize features
        X = self.scaler.transform(data[feature_columns])
        
        # Make predictions
        predictions = self.model.predict(X)
        
        return pd.Series(predictions, index=data.index)
    
    def crossvalidate_model(self, df, target_column='direction', prediction_horizon=5, n_splits=5):
        """
        Perform time series cross validation
        
        Args:
            df: DataFrame with technical indicators
            target_column: Column to predict ('direction', 'close', 'returns')
            prediction_horizon: How many days into the future to predict
            n_splits: Number of cross-validation splits
        
        Returns:
            Dictionary with cross-validation metrics
        """
        # Create a copy of the dataframe
        data = df.copy()
        
        # Create target variable (future price/return)
        if target_column == 'direction':
            # Binary classification: price direction (up/down)
            data['target'] = np.where(data['close'].shift(-prediction_horizon) > data['close'], 1, 0)
        elif target_column == 'returns':
            # Regression: future returns
            data['target'] = data['close'].pct_change(prediction_horizon).shift(-prediction_horizon)
        else:
            # Regression: future prices
            data['target'] = data[target_column].shift(-prediction_horizon)
        
        # Drop rows with NaN in target
        data = data.dropna(subset=['target'])
        
        # Define features to use (all technical indicators)
        # Exclude non-feature columns
        exclude_columns = ['target', 'open', 'high', 'low', 'close', 'volume',
                          'returns', 'cumulative_returns', 'drawdown',
                          'position', 'cash', 'holdings', 'portfolio_value',
                          'trade_type', 'trade_price', 'trade_shares']
        
        # Filter columns to include only numeric ones
        feature_columns = [col for col in data.columns 
                          if col not in exclude_columns 
                          and np.issubdtype(data[col].dtype, np.number)]
        
        # Handle NaN values in features
        data = data.dropna(subset=feature_columns)
        
        # Split data for time series cross validation
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        X = data[feature_columns]
        y = data['target']
        
        # Initialize metrics
        if target_column == 'direction':
            # Classification metrics
            metrics = {
                'accuracy': [],
                'precision': [],
                'recall': [],
                'f1_score': []
            }
        else:
            # Regression metrics
            metrics = {
                'mse': [],
                'rmse': [],
                'r2_score': []
            }
        
        # Perform cross validation
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Standardize features
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            if target_column == 'direction':
                # Classification
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                
                metrics['accuracy'].append(accuracy_score(y_test, y_pred))
                metrics['precision'].append(precision_score(y_test, y_pred))
                metrics['recall'].append(recall_score(y_test, y_pred))
                metrics['f1_score'].append(f1_score(y_test, y_pred))
            else:
                # Regression
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                
                mse = mean_squared_error(y_test, y_pred)
                metrics['mse'].append(mse)
                metrics['rmse'].append(np.sqrt(mse))
                metrics['r2_score'].append(r2_score(y_test, y_pred))
        
        # Calculate average metrics
        avg_metrics = {metric: np.mean(values) for metric, values in metrics.items()}
        std_metrics = {f"{metric}_std": np.std(values) for metric, values in metrics.items()}
        
        return {**avg_metrics, **std_metrics}
    
    def generate_trading_signals(self, df, prediction_column='prediction', threshold=0.5):
        """
        Generate trading signals based on ML predictions
        
        Args:
            df: DataFrame with predictions
            prediction_column: Column with predictions
            threshold: Threshold for generating signals (for classification)
        
        Returns:
            DataFrame with trading signals
        """
        # Create a copy of the dataframe
        result = df.copy()
        
        # Create signal column
        result['ml_signal'] = 0
        
        # Check whether prediction is classification or regression
        if prediction_column in result.columns:
            unique_values = result[prediction_column].dropna().unique()
            if len(unique_values) <= 2 and set(unique_values).issubset({0, 1}):
                # Classification predictions
                result.loc[result[prediction_column] >= threshold, 'ml_signal'] = 1
                result.loc[result[prediction_column] < threshold, 'ml_signal'] = -1
            else:
                # Regression predictions (change in price)
                result.loc[result[prediction_column] > 0, 'ml_signal'] = 1
                result.loc[result[prediction_column] < 0, 'ml_signal'] = -1
        
        return result