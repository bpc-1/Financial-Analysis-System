# signal_generator.py - Trading signal generation

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class SignalGenerator:
    """Class for generating trading signals based on technical analysis"""
    
    def generate_moving_average_signals(self, df, fast_ma='sma_20', slow_ma='sma_50'):
        """Generate signals based on moving average crossovers"""
        result = df.copy()
        
        # Check if required columns exist
        if fast_ma not in df.columns or slow_ma not in df.columns:
            missing = []
            if fast_ma not in df.columns:
                missing.append(fast_ma)
            if slow_ma not in df.columns:
                missing.append(slow_ma)
            logger.warning(f"Missing columns for MA signals: {missing}. Available: {df.columns.tolist()}")
            
            # Create signal column but set all to 0
            result['ma_signal'] = 0
            return result
        
        # Create signal column (1 = buy, -1 = sell, 0 = hold)
        result['ma_signal'] = 0
        
        # Generate buy signal when fast MA crosses above slow MA
        result.loc[result[fast_ma] > result[slow_ma], 'ma_signal'] = 1
        
        # Generate sell signal when fast MA crosses below slow MA
        result.loc[result[fast_ma] < result[slow_ma], 'ma_signal'] = -1
        
        return result
    
    def generate_macd_signals(self, df):
        """Generate signals based on MACD crossovers"""
        result = df.copy()
        
        # Check if required columns exist
        if 'macd' not in df.columns or 'macd_signal' not in df.columns:
            # Create signal column but set all to 0
            result['macd_signal_indicator'] = 0
            return result
        
        # Create signal column
        result['macd_signal_indicator'] = 0
        
        # Buy signal when MACD crosses above signal line
        result.loc[result['macd'] > result['macd_signal'], 'macd_signal_indicator'] = 1
        
        # Sell signal when MACD crosses below signal line
        result.loc[result['macd'] < result['macd_signal'], 'macd_signal_indicator'] = -1
        
        return result
    
    def generate_rsi_signals(self, df, overbought=70, oversold=30):
        """Generate signals based on RSI overbought/oversold levels"""
        result = df.copy()
        
        # Check if required columns exist
        if 'rsi' not in df.columns:
            # Create signal column but set all to 0
            result['rsi_signal'] = 0
            return result
        
        # Create signal column
        result['rsi_signal'] = 0
        
        # Buy signal when RSI crosses above oversold level
        result.loc[result['rsi'] < oversold, 'rsi_signal'] = 1
        
        # Sell signal when RSI crosses below overbought level
        result.loc[result['rsi'] > overbought, 'rsi_signal'] = -1
        
        return result
    
    def generate_bollinger_signals(self, df):
        """Generate signals based on Bollinger Band bounces"""
        result = df.copy()
        
        # Check if required columns exist
        if 'bollinger_upper' not in df.columns or 'bollinger_lower' not in df.columns:
            # Create signal column but set all to 0
            result['bollinger_signal'] = 0
            return result
        
        # Create signal column
        result['bollinger_signal'] = 0
        
        # Buy signal when price touches lower band
        result.loc[result['close'] <= result['bollinger_lower'], 'bollinger_signal'] = 1
        
        # Sell signal when price touches upper band
        result.loc[result['close'] >= result['bollinger_upper'], 'bollinger_signal'] = -1
        
        return result
    
    def generate_stochastic_signals(self, df, overbought=80, oversold=20):
        """Generate signals based on Stochastic Oscillator"""
        result = df.copy()
        
        # Check if required columns exist
        if 'stoch_k' not in df.columns or 'stoch_d' not in df.columns:
            # Create signal column but set all to 0
            result['stoch_signal'] = 0
            return result
        
        # Create signal column
        result['stoch_signal'] = 0
        
        # Buy signal when %K crosses above %D in oversold territory
        result.loc[(result['stoch_k'] > result['stoch_d']) & 
                   (result['stoch_k'] < oversold) & 
                   (result['stoch_d'] < oversold), 'stoch_signal'] = 1
        
        # Sell signal when %K crosses below %D in overbought territory
        result.loc[(result['stoch_k'] < result['stoch_d']) & 
                   (result['stoch_k'] > overbought) & 
                   (result['stoch_d'] > overbought), 'stoch_signal'] = -1
        
        return result
    
    def generate_adx_signals(self, df, threshold=25):
        """Generate signals based on ADX strength"""
        result = df.copy()
        
        # Check if required columns exist
        if 'adx' not in df.columns or '+di' not in df.columns or '-di' not in df.columns:
            # Create signal column but set all to 0
            result['adx_signal'] = 0
            return result
        
        # Create signal column
        result['adx_signal'] = 0
        
        # Buy signal when ADX > threshold and +DI > -DI
        result.loc[(result['adx'] > threshold) & 
                  (result['+di'] > result['-di']), 'adx_signal'] = 1
        
        # Sell signal when ADX > threshold and +DI < -DI
        result.loc[(result['adx'] > threshold) & 
                  (result['+di'] < result['-di']), 'adx_signal'] = -1
        
        return result
    
    def generate_obv_signals(self, df):
        """Generate signals based on On-Balance Volume"""
        result = df.copy()
        
        # Check if required columns exist
        if 'obv' not in df.columns or 'obv_ema' not in df.columns:
            # Create signal column but set all to 0
            result['obv_signal'] = 0
            return result
        
        # Create signal column
        result['obv_signal'] = 0
        
        # Buy signal when OBV crosses above its EMA
        result.loc[result['obv'] > result['obv_ema'], 'obv_signal'] = 1
        
        # Sell signal when OBV crosses below its EMA
        result.loc[result['obv'] < result['obv_ema'], 'obv_signal'] = -1
        
        return result
    
    def generate_psar_signals(self, df):
        """Generate signals based on Parabolic SAR"""
        result = df.copy()
        
        # Check if required columns exist
        if 'psar' not in df.columns or 'psar_trend' not in df.columns:
            # Create signal column but set all to 0
            result['psar_signal'] = 0
            return result
        
        # Create signal column
        result['psar_signal'] = 0
        
        # Generate signals based on trend reversals
        result['psar_signal'] = result['psar_trend'].diff()
        
        return result
    
    def generate_ichimoku_signals(self, df):
        """Generate signals based on Ichimoku Cloud"""
        result = df.copy()
        
        # Check if required columns exist
        required_cols = ['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b']
        if not all(col in df.columns for col in required_cols):
            # Create signal column but set all to 0
            result['ichimoku_signal'] = 0
            return result
        
        # Create signal column
        result['ichimoku_signal'] = 0
        
        # Buy signal when:
        # 1. Tenkan-sen crosses above Kijun-sen (TK Cross)
        # 2. Price is above the cloud (Senkou Span A & B)
        result.loc[(result['tenkan_sen'] > result['kijun_sen']) & 
                   (result['close'] > result['senkou_span_a']) & 
                   (result['close'] > result['senkou_span_b']), 'ichimoku_signal'] = 1
        
        # Sell signal when:
        # 1. Tenkan-sen crosses below Kijun-sen
        # 2. Price is below the cloud
        result.loc[(result['tenkan_sen'] < result['kijun_sen']) & 
                   (result['close'] < result['senkou_span_a']) & 
                   (result['close'] < result['senkou_span_b']), 'ichimoku_signal'] = -1
        
        return result
    
    def generate_volume_signals(self, df):
        """Generate signals based on volume indicators"""
        result = df.copy()
        
        # Check if required columns exist
        if 'volume' not in df.columns or 'volume_sma_20' not in df.columns:
            # Create signal column but set all to 0
            result['volume_signal'] = 0
            return result
        
        # Create signal column
        result['volume_signal'] = 0
        
        # Generate signals
        # Buy signal when volume is significantly higher than its moving average during price increase
        result.loc[(result['volume'] > 1.5 * result['volume_sma_20']) & 
                   (result['close'] > result['close'].shift(1)), 'volume_signal'] = 1
        
        # Sell signal when volume is significantly higher than its moving average during price decrease
        result.loc[(result['volume'] > 1.5 * result['volume_sma_20']) & 
                   (result['close'] < result['close'].shift(1)), 'volume_signal'] = -1
        
        return result
    
    def generate_sentiment_signals(self, df):
        """Generate signals based on sentiment data"""
        result = df.copy()
        
        # Check if required columns exist
        if 'sentiment_score' not in df.columns:
            # Create signal column but set all to 0
            result['sentiment_signal'] = 0
            return result
        
        # Create signal column
        result['sentiment_signal'] = 0
        
        # Buy signal when sentiment is strongly positive
        result.loc[result['sentiment_score'] > 0.5, 'sentiment_signal'] = 1
        
        # Sell signal when sentiment is strongly negative
        result.loc[result['sentiment_score'] < -0.5, 'sentiment_signal'] = -1
        
        return result
    
    def generate_combined_signal(self, df, weights=None):
        """Generate a combined signal based on multiple indicators with weights"""
        if weights is None:
            weights = {
                'ma_signal': 0.15,
                'macd_signal_indicator': 0.15,
                'rsi_signal': 0.1,
                'bollinger_signal': 0.1,
                'stoch_signal': 0.1,
                'adx_signal': 0.1,
                'obv_signal': 0.05,
                'psar_signal': 0.05,
                'ichimoku_signal': 0.1,
                'volume_signal': 0.05,
                'sentiment_signal': 0.05
            }
        
        result = df.copy()
        
        # Generate individual signals
        result = self.generate_moving_average_signals(result)
        result = self.generate_macd_signals(result)
        result = self.generate_rsi_signals(result)
        result = self.generate_bollinger_signals(result)
        result = self.generate_stochastic_signals(result)
        result = self.generate_adx_signals(result)
        result = self.generate_obv_signals(result)
        result = self.generate_psar_signals(result)
        result = self.generate_ichimoku_signals(result)
        result = self.generate_volume_signals(result)
        result = self.generate_sentiment_signals(result)
        
        # Calculate weighted signal
        result['combined_signal'] = 0
        for signal, weight in weights.items():
            if signal in result.columns:
                result['combined_signal'] += result[signal] * weight
        
        # Threshold the combined signal
        result['final_signal'] = 0
        result.loc[result['combined_signal'] > 0.3, 'final_signal'] = 1
        result.loc[result['combined_signal'] < -0.3, 'final_signal'] = -1
        
        return result