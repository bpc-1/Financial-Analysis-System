# technical_analysis.py - Enhanced technical analysis

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class TechnicalAnalyzer:
    """Class for performing technical analysis on financial data"""
    
    def calculate_moving_averages(self, df, windows=[20, 50, 200]):
        """Calculate simple moving averages for the given windows"""
        result = df.copy()
        
        # Make sure we have a 'close' column
        if 'close' not in df.columns:
            logger.error(f"Missing 'close' column in dataframe. Columns: {df.columns.tolist()}")
            raise ValueError("DataFrame must have a 'close' column")
            
        for window in windows:
            if len(df) >= window:  # Only calculate if we have enough data
                result[f'sma_{window}'] = df['close'].rolling(window=window).mean()
        return result
    
    def calculate_exponential_moving_averages(self, df, windows=[12, 26, 50]):
        """Calculate exponential moving averages for the given windows"""
        result = df.copy()
        for window in windows:
            if len(df) >= window:  # Only calculate if we have enough data
                result[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
        return result
    
    def calculate_macd(self, df, fast=12, slow=26, signal=9):
        """Calculate MACD (Moving Average Convergence Divergence)"""
        result = df.copy()
        
        # Only calculate if we have enough data
        if len(df) >= slow:
            result['ema_fast'] = df['close'].ewm(span=fast, adjust=False).mean()
            result['ema_slow'] = df['close'].ewm(span=slow, adjust=False).mean()
            result['macd'] = result['ema_fast'] - result['ema_slow']
            
            if len(df) >= slow + signal:
                result['macd_signal'] = result['macd'].ewm(span=signal, adjust=False).mean()
                result['macd_histogram'] = result['macd'] - result['macd_signal']
        
        return result
    
    def calculate_rsi(self, df, window=14):
        """Calculate RSI (Relative Strength Index)"""
        result = df.copy()
        
        # Only calculate if we have enough data
        if len(df) >= window + 1:
            delta = df['close'].diff()
            
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=window).mean()
            avg_loss = loss.rolling(window=window).mean()
            
            # Avoid division by zero
            avg_loss = avg_loss.replace(0, 0.000001)
            
            # Calculate RSI
            rs = avg_gain / avg_loss
            result['rsi'] = 100 - (100 / (1 + rs))
        
        return result
    
    def calculate_bollinger_bands(self, df, window=20, num_std=2):
        """Calculate Bollinger Bands"""
        result = df.copy()
        
        # Only calculate if we have enough data
        if len(df) >= window:
            result['sma'] = df['close'].rolling(window=window).mean()
            result['std'] = df['close'].rolling(window=window).std()
            result['bollinger_upper'] = result['sma'] + (result['std'] * num_std)
            result['bollinger_lower'] = result['sma'] - (result['std'] * num_std)
        
        return result
    
    def calculate_stochastic_oscillator(self, df, k_period=14, d_period=3):
        """Calculate Stochastic Oscillator"""
        result = df.copy()
        
        # Only calculate if we have enough data
        if len(df) >= k_period:
            # Calculate %K
            lowest_low = df['low'].rolling(window=k_period).min()
            highest_high = df['high'].rolling(window=k_period).max()
            
            # Avoid division by zero
            denominator = highest_high - lowest_low
            denominator = denominator.replace(0, 0.000001)
            
            result['stoch_k'] = 100 * ((df['close'] - lowest_low) / denominator)
            
            # Calculate %D (simple moving average of %K)
            if len(df) >= k_period + d_period - 1:
                result['stoch_d'] = result['stoch_k'].rolling(window=d_period).mean()
        
        return result
    
    def calculate_average_directional_index(self, df, window=14):
        """Calculate Average Directional Index (ADX)"""
        result = df.copy()
        
        # Only calculate if we have enough data
        if len(df) >= window + 1:
            # Calculate True Range (TR)
            result['high_low'] = df['high'] - df['low']
            result['high_close'] = abs(df['high'] - df['close'].shift(1))
            result['low_close'] = abs(df['low'] - df['close'].shift(1))
            result['tr'] = result[['high_low', 'high_close', 'low_close']].max(axis=1)
            
            # Calculate Directional Movement (DM)
            result['up_move'] = df['high'].diff()
            result['down_move'] = df['low'].diff()
            
            # Positive Directional Movement (+DM)
            result['+dm'] = np.where(
                (result['up_move'] > result['down_move']) & (result['up_move'] > 0),
                result['up_move'],
                0
            )
            
            # Negative Directional Movement (-DM)
            result['-dm'] = np.where(
                (result['down_move'] > result['up_move']) & (result['down_move'] > 0),
                result['down_move'],
                0
            )
            
            # Calculate Smoothed Moving Averages of TR, +DM, -DM
            result['tr_ema'] = result['tr'].ewm(span=window, min_periods=window).mean()
            result['+dm_ema'] = result['+dm'].ewm(span=window, min_periods=window).mean()
            result['-dm_ema'] = result['-dm'].ewm(span=window, min_periods=window).mean()
            
            # Calculate Directional Indicators
            result['+di'] = 100 * result['+dm_ema'] / result['tr_ema']
            result['-di'] = 100 * result['-dm_ema'] / result['tr_ema']
            
            # Calculate Directional Index (DX)
            result['dx'] = 100 * abs(result['+di'] - result['-di']) / (result['+di'] + result['-di'])
            
            # Calculate Average Directional Index (ADX)
            result['adx'] = result['dx'].ewm(span=window, min_periods=window).mean()
            
            # Clean up intermediate columns
            result = result.drop(['high_low', 'high_close', 'low_close', 'tr', 'up_move', 
                                 'down_move', '+dm', '-dm', 'tr_ema', '+dm_ema', '-dm_ema', 'dx'], axis=1)
        
        return result
    
    def calculate_on_balance_volume(self, df):
        """Calculate On-Balance Volume (OBV)"""
        result = df.copy()
        
        # Initialize OBV with the first row's volume
        result['obv'] = 0
        
        # Calculate OBV for the rest of the rows
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                result['obv'].iloc[i] = result['obv'].iloc[i-1] + df['volume'].iloc[i]
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                result['obv'].iloc[i] = result['obv'].iloc[i-1] - df['volume'].iloc[i]
            else:
                result['obv'].iloc[i] = result['obv'].iloc[i-1]
        
        # Calculate OBV EMA
        if len(df) >= 20:
            result['obv_ema'] = result['obv'].ewm(span=20).mean()
        
        return result
    
    def calculate_parabolic_sar(self, df, step=0.02, max_step=0.2):
        """Calculate Parabolic SAR"""
        result = df.copy()
        
        # Need at least 2 periods to calculate
        if len(df) < 2:
            return result
        
        # Initialize variables
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        sar = [0] * len(df)
        trend = [0] * len(df)  # 1 for uptrend, -1 for downtrend
        ep = [0] * len(df)  # Extreme point
        af = [step] * len(df)  # Acceleration factor
        
        # Initialize first SAR, trend, and extreme point
        trend[0] = 1 if close[1] > close[0] else -1
        sar[0] = low[0] if trend[0] == 1 else high[0]
        ep[0] = high[0] if trend[0] == 1 else low[0]
        
        # Calculate SAR for each period
        for i in range(1, len(df)):
            # Update SAR
            sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
            
            # Ensure SAR doesn't go beyond price range
            if trend[i-1] == 1:  # Uptrend
                sar[i] = min(sar[i], low[i-1], low[i-2] if i > 1 else low[i-1])
            else:  # Downtrend
                sar[i] = max(sar[i], high[i-1], high[i-2] if i > 1 else high[i-1])
            
            # Check for trend reversal
            if trend[i-1] == 1 and low[i] < sar[i]:
                trend[i] = -1  # Switch to downtrend
                sar[i] = max(high[i], high[i-1])
                ep[i] = low[i]
                af[i] = step
            elif trend[i-1] == -1 and high[i] > sar[i]:
                trend[i] = 1  # Switch to uptrend
                sar[i] = min(low[i], low[i-1])
                ep[i] = high[i]
                af[i] = step
            else:
                trend[i] = trend[i-1]  # Continue current trend
                
                # Update extreme point
                if trend[i] == 1 and high[i] > ep[i-1]:
                    ep[i] = high[i]
                    af[i] = min(af[i-1] + step, max_step)
                elif trend[i] == -1 and low[i] < ep[i-1]:
                    ep[i] = low[i]
                    af[i] = min(af[i-1] + step, max_step)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
        
        # Add results to DataFrame
        result['psar'] = sar
        result['psar_trend'] = trend
        
        return result
    
    def calculate_ichimoku_cloud(self, df, conversion_period=9, base_period=26, lagging_span2_period=52, displacement=26):
        """Calculate Ichimoku Cloud components"""
        result = df.copy()
        
        # Tenkan-sen (Conversion Line): (highest high + lowest low) / 2 for the conversion_period
        if len(df) >= conversion_period:
            result['tenkan_sen'] = (
                df['high'].rolling(window=conversion_period).max() + 
                df['low'].rolling(window=conversion_period).min()
            ) / 2
        
        # Kijun-sen (Base Line): (highest high + lowest low) / 2 for the base_period
        if len(df) >= base_period:
            result['kijun_sen'] = (
                df['high'].rolling(window=base_period).max() + 
                df['low'].rolling(window=base_period).min()
            ) / 2
        
        # Senkou Span A (Leading Span A): (Conversion Line + Base Line) / 2, shifted forward by displacement periods
        if len(df) >= max(conversion_period, base_period):
            result['senkou_span_a'] = ((result['tenkan_sen'] + result['kijun_sen']) / 2).shift(displacement)
        
        # Senkou Span B (Leading Span B): (highest high + lowest low) / 2 for lagging_span2_period, shifted forward by displacement periods
        if len(df) >= lagging_span2_period:
            result['senkou_span_b'] = (
                (df['high'].rolling(window=lagging_span2_period).max() + 
                 df['low'].rolling(window=lagging_span2_period).min()) / 2
            ).shift(displacement)
        
        # Chikou Span (Lagging Span): Close price, shifted backwards by displacement periods
        result['chikou_span'] = df['close'].shift(-displacement)
        
        return result
    
    def calculate_fibonacci_levels(self, df, window=100):
        """Calculate Fibonacci retracement levels based on recent high-low range"""
        result = df.copy()
        
        if len(df) < window:
            return result
        
        # Find highest high and lowest low in the window
        window_high = df['high'].rolling(window=window).max()
        window_low = df['low'].rolling(window=window).min()
        
        # Calculate Fibonacci levels
        range_price = window_high - window_low
        result['fib_0'] = window_low  # 0% retracement (the low)
        result['fib_23_6'] = window_low + range_price * 0.236  # 23.6% retracement
        result['fib_38_2'] = window_low + range_price * 0.382  # 38.2% retracement
        result['fib_50_0'] = window_low + range_price * 0.5    # 50.0% retracement
        result['fib_61_8'] = window_low + range_price * 0.618  # 61.8% retracement
        result['fib_78_6'] = window_low + range_price * 0.786  # 78.6% retracement
        result['fib_100'] = window_high  # 100% retracement (the high)
        
        return result
    
    def calculate_volume_indicators(self, df):
        """Calculate volume-based indicators"""
        result = df.copy()
        
        # Volume SMA
        if len(df) >= 20:
            result['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        
        # Money Flow Index (MFI)
        if len(df) >= 14:
            # Calculate typical price
            result['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
            
            # Calculate raw money flow
            result['money_flow'] = result['typical_price'] * df['volume']
            
            # Calculate positive and negative money flow
            result['positive_flow'] = np.where(
                result['typical_price'] > result['typical_price'].shift(1),
                result['money_flow'],
                0
            )
            
            result['negative_flow'] = np.where(
                result['typical_price'] < result['typical_price'].shift(1),
                result['money_flow'],
                0
            )
            
            # Calculate 14-period positive and negative money flow sum
            result['positive_flow_sum'] = result['positive_flow'].rolling(window=14).sum()
            result['negative_flow_sum'] = result['negative_flow'].rolling(window=14).sum()
            
            # Calculate money ratio and MFI
            result['money_ratio'] = result['positive_flow_sum'] / result['negative_flow_sum'].replace(0, 0.000001)
            result['mfi'] = 100 - (100 / (1 + result['money_ratio']))
            
            # Clean up intermediate columns
            result = result.drop(['typical_price', 'money_flow', 'positive_flow', 'negative_flow', 
                                 'positive_flow_sum', 'negative_flow_sum', 'money_ratio'], axis=1)
        
        return result
    
    def analyze_ticker(self, df):
        """Perform comprehensive technical analysis on a ticker"""
        # Verify we have the necessary data
        if 'close' not in df.columns:
            raise ValueError(f"DataFrame must have a 'close' column. Available columns: {df.columns.tolist()}")
        
        if len(df) < 20:  # Need at least some data for analysis
            raise ValueError(f"Not enough data points for analysis. Got {len(df)}, need at least 20.")
        
        result = df.copy()
        
        # Calculate moving averages
        result = self.calculate_moving_averages(result)
        result = self.calculate_exponential_moving_averages(result)
        
        # Calculate oscillators
        result = self.calculate_macd(result)
        result = self.calculate_rsi(result)
        
        # Calculate Bollinger Bands
        result = self.calculate_bollinger_bands(result)
        
        # Calculate additional indicators
        result = self.calculate_stochastic_oscillator(result)
        result = self.calculate_average_directional_index(result)
        result = self.calculate_on_balance_volume(result)
        result = self.calculate_parabolic_sar(result)
        result = self.calculate_ichimoku_cloud(result)
        result = self.calculate_fibonacci_levels(result)
        result = self.calculate_volume_indicators(result)
        
        return result
    
    def combine_with_sentiment(self, technical_df, sentiment_df):
        """Combine technical analysis with sentiment data"""
        if sentiment_df is None or sentiment_df.empty:
            return technical_df
        
        # Make sure both DataFrames have the same index
        combined_df = technical_df.copy()
        
        # Merge sentiment data
        sentiment_df = sentiment_df.reindex(combined_df.index, method='ffill')  # Forward fill missing values
        
        # Add sentiment columns
        if 'sentiment_score' in sentiment_df.columns:
            combined_df['sentiment_score'] = sentiment_df['sentiment_score']
        
        if 'volume' in sentiment_df.columns:
            combined_df['sentiment_volume'] = sentiment_df['volume']
        
        # Calculate a sentiment-weighted close indicator
        if 'sentiment_score' in combined_df.columns:
            # Scale sentiment from [-1, 1] to [0.5, 1.5] to use as a multiplier
            sentiment_multiplier = 1 + combined_df['sentiment_score'] / 2
            combined_df['sentiment_adjusted_close'] = combined_df['close'] * sentiment_multiplier
        
        return combined_df