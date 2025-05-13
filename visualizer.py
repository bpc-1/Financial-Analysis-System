# visualizer.py - Visualization components

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import logging
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

logger = logging.getLogger(__name__)

class Visualizer:
    """Class for visualizing financial data and analysis results"""
    
    def __init__(self, style='dark_background'):
        """Initialize visualizer with a style"""
        self.style = style
        plt.style.use(style)
    
    def plot_price_with_indicators(self, df, title="Price Chart with Indicators", figsize=(14, 10)):
        """Plot price data with technical indicators"""
        plt.figure(figsize=figsize)
        
        # Verify we have required columns
        if 'close' not in df.columns:
            raise ValueError("DataFrame must have a 'close' column for visualization")
        
        # Create subplots grid
        n_rows = 5  # Price, MACD, RSI, Stochastic, Volume
        gs = plt.GridSpec(n_rows, 1, height_ratios=[3, 1, 1, 1, 1])
        
        # Plot price
        ax1 = plt.subplot(gs[0])
        ax1.plot(df.index, df['close'], label='Price', linewidth=1.5)
        
        # Plot moving averages if available
        for ma in [20, 50, 200]:
            col = f'sma_{ma}'
            if col in df.columns:
                ax1.plot(df.index, df[col], label=f'SMA {ma}', alpha=0.7)
        
        # Plot Bollinger Bands if available
        if 'bollinger_upper' in df.columns and 'bollinger_lower' in df.columns:
            ax1.plot(df.index, df['bollinger_upper'], 'k--', alpha=0.3)
            ax1.plot(df.index, df['bollinger_lower'], 'k--', alpha=0.3)
            ax1.fill_between(df.index, df['bollinger_lower'], df['bollinger_upper'], color='gray', alpha=0.1)
        
        # Plot Ichimoku Cloud if available
        if all(col in df.columns for col in ['senkou_span_a', 'senkou_span_b']):
            # Fill between Senkou Span A and B (the cloud)
            green_cloud = df.index[df['senkou_span_a'] >= df['senkou_span_b']]
            red_cloud = df.index[df['senkou_span_a'] < df['senkou_span_b']]
            
            # Plot green cloud (bullish)
            if len(green_cloud) > 0:
                green_df = df.loc[green_cloud]
                ax1.fill_between(green_cloud, green_df['senkou_span_a'], green_df['senkou_span_b'], 
                              color='green', alpha=0.1)
            
            # Plot red cloud (bearish)
            if len(red_cloud) > 0:
                red_df = df.loc[red_cloud]
                ax1.fill_between(red_cloud, red_df['senkou_span_a'], red_df['senkou_span_b'], 
                              color='red', alpha=0.1)
            
            # Plot Ichimoku lines
            if 'tenkan_sen' in df.columns:
                ax1.plot(df.index, df['tenkan_sen'], color='blue', linestyle='-', 
                      linewidth=0.7, alpha=0.7, label='Tenkan-sen')
            
            if 'kijun_sen' in df.columns:
                ax1.plot(df.index, df['kijun_sen'], color='maroon', linestyle='-', 
                      linewidth=0.7, alpha=0.7, label='Kijun-sen')
        
        # Add buy/sell signals if available
        if 'final_signal' in df.columns:
            buy_signals = df[df['final_signal'] == 1]
            sell_signals = df[df['final_signal'] == -1]
            
            if not buy_signals.empty:
                ax1.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', s=100, label='Buy Signal')
            if not sell_signals.empty:
                ax1.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', s=100, label='Sell Signal')
        
        ax1.set_title(title)
        ax1.set_ylabel('Price')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Format date axis
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax1.get_xticklabels(), rotation=30, ha='right')
        
        # Plot MACD if available
        if all(col in df.columns for col in ['macd', 'macd_signal']):
            ax2 = plt.subplot(gs[1], sharex=ax1)
            ax2.plot(df.index, df['macd'], label='MACD', color='blue', linewidth=1.0)
            ax2.plot(df.index, df['macd_signal'], label='Signal Line', color='red', linewidth=1.0)
            
            # Plot histogram with colors
            if 'macd_histogram' in df.columns:
                # Color positive and negative bars differently
                positive = df['macd_histogram'] > 0
                negative = df['macd_histogram'] <= 0
                
                ax2.bar(df.index[positive], df['macd_histogram'][positive], color='green', alpha=0.5, width=1)
                ax2.bar(df.index[negative], df['macd_histogram'][negative], color='red', alpha=0.5, width=1)
            
            ax2.set_ylabel('MACD')
            ax2.legend(loc='upper left')
            ax2.grid(True, alpha=0.3)
        
        # Plot RSI if available
        if 'rsi' in df.columns:
            ax3 = plt.subplot(gs[2], sharex=ax1)
            ax3.plot(df.index, df['rsi'], label='RSI', color='purple', linewidth=1.0)
            ax3.axhline(70, color='red', linestyle='--', alpha=0.5)
            ax3.axhline(30, color='green', linestyle='--', alpha=0.5)
            ax3.fill_between(df.index, 70, df['rsi'].where(df['rsi'] > 70), color='red', alpha=0.2)
            ax3.fill_between(df.index, 30, df['rsi'].where(df['rsi'] < 30), color='green', alpha=0.2)
            ax3.set_ylim(0, 100)
            ax3.set_ylabel('RSI')
            ax3.legend(loc='upper left')
            ax3.grid(True, alpha=0.3)
        
        # Plot Stochastic Oscillator if available
        if 'stoch_k' in df.columns and 'stoch_d' in df.columns:
            ax4 = plt.subplot(gs[3], sharex=ax1)
            ax4.plot(df.index, df['stoch_k'], label='%K', color='blue', linewidth=1.0)
            ax4.plot(df.index, df['stoch_d'], label='%D', color='red', linewidth=1.0)
            ax4.axhline(80, color='red', linestyle='--', alpha=0.5)
            ax4.axhline(20, color='green', linestyle='--', alpha=0.5)
            ax4.fill_between(df.index, 80, df['stoch_k'].where(df['stoch_k'] > 80), color='red', alpha=0.2)
            ax4.fill_between(df.index, 20, df['stoch_k'].where(df['stoch_k'] < 20), color='green', alpha=0.2)
            ax4.set_ylim(0, 100)
            ax4.set_ylabel('Stochastic')
            ax4.legend(loc='upper left')
            ax4.grid(True, alpha=0.3)
        
        # Plot Volume if available
        if 'volume' in df.columns:
            ax5 = plt.subplot(gs[4], sharex=ax1)
            
            # Plot volume bars with colors based on price change
            price_change = df['close'].diff()
            positive = price_change > 0
            negative = price_change <= 0
            
            if 'volume_sma_20' in df.columns:
                ax5.plot(df.index, df['volume_sma_20'], color='blue', linewidth=1.0, label='Volume SMA 20')
            
            ax5.bar(df.index[positive], df['volume'][positive], color='green', alpha=0.5, width=1)
            ax5.bar(df.index[negative], df['volume'][negative], color='red', alpha=0.5, width=1)
            
            ax5.set_ylabel('Volume')
            if 'volume_sma_20' in df.columns:
                ax5.legend(loc='upper left')
            ax5.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return plt.gcf()
    
    def plot_backtest_results(self, df, metrics, title="Backtest Results", figsize=(14, 8)):
        """Plot backtest results"""
        plt.figure(figsize=figsize)
        
        # Verify we have required columns
        if 'portfolio_value' not in df.columns:
            raise ValueError("DataFrame must have a 'portfolio_value' column for backtest visualization")
        
        # Create subplots grid
        gs = plt.GridSpec(3, 1, height_ratios=[2, 1, 1])
        
        # Plot portfolio value
        ax1 = plt.subplot(gs[0])
        ax1.plot(df.index, df['portfolio_value'], label='Portfolio Value', linewidth=1.5)
        
        # Add buy/sell markers if available
        if 'position' in df.columns and 'cash' in df.columns:
            # Identify points where position changes
            position_changes = df['position'].diff().abs() > 0
            trades = df[position_changes]
            
            # Identify buy and sell points
            buy_points = trades[trades['position'] > trades['position'].shift(1).fillna(0)]
            sell_points = trades[trades['position'] < trades['position'].shift(1).fillna(0)]
            
            if not buy_points.empty:
                ax1.scatter(buy_points.index, buy_points['portfolio_value'], 
                         marker='^', color='green', s=100, label='Buy')
            
            if not sell_points.empty:
                ax1.scatter(sell_points.index, sell_points['portfolio_value'], 
                         marker='v', color='red', s=100, label='Sell')
        
        # Add a horizontal line for initial capital
        initial_capital = df['portfolio_value'].iloc[0]
        ax1.axhline(initial_capital, color='gray', linestyle='--', alpha=0.5, 
                  label=f'Initial Capital (${initial_capital:,.2f})')
        
        # Add metrics as text box
        metrics_text = f"Total Return: {metrics['total_return']:.2%}\n"
        metrics_text += f"Annualized Return: {metrics['annualized_return']:.2%}\n"
        metrics_text += f"Max Drawdown: {metrics['max_drawdown']:.2%}\n"
        metrics_text += f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}"
        
        if 'win_rate' in metrics:
            metrics_text += f"\nWin Rate: {metrics['win_rate']:.2%}"
        
        if 'num_trades' in metrics:
            metrics_text += f"\nNumber of Trades: {metrics['num_trades']}"
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax1.text(0.05, 0.95, metrics_text, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        ax1.set_title(title)
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Format date axis
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax1.get_xticklabels(), rotation=30, ha='right')
        
        # Plot returns
        if 'returns' in df.columns:
            ax2 = plt.subplot(gs[1], sharex=ax1)
            
            # Plot returns with colors
            positive_returns = df['returns'] > 0
            negative_returns = df['returns'] <= 0
            
            ax2.bar(df.index[positive_returns], df['returns'][positive_returns], 
                  color='green', alpha=0.5, width=1, label='Positive Returns')
            ax2.bar(df.index[negative_returns], df['returns'][negative_returns], 
                  color='red', alpha=0.5, width=1, label='Negative Returns')
            
            ax2.set_ylabel('Daily Returns')
            ax2.legend(loc='upper left')
            ax2.grid(True, alpha=0.3)
        
        # Plot drawdown
        if 'drawdown' in df.columns:
            ax3 = plt.subplot(gs[2], sharex=ax1)
            ax3.fill_between(df.index, df['drawdown'] * 100, 0, color='red', alpha=0.3)
            ax3.plot(df.index, df['drawdown'] * 100, color='red', alpha=0.5)
            
            # Mark maximum drawdown
            max_dd_idx = df['drawdown'].idxmax()
            max_dd = df['drawdown'].max()
            ax3.scatter(max_dd_idx, max_dd * 100, color='darkred', s=80, 
                     marker='o', label=f'Max DD: {max_dd:.2%}')
            
            ax3.set_ylabel('Drawdown (%)')
            ax3.set_ylim(0, max(100, df['drawdown'].max() * 100 * 1.1))  # Add 10% margin
            ax3.legend(loc='upper left')
            ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return plt.gcf()
    
    def plot_portfolio_allocation(self, weights, title="Portfolio Allocation", figsize=(10, 6)):
        """Plot portfolio allocation as a pie chart"""
        plt.figure(figsize=figsize)
        
        # Sort weights by value (descending)
        sorted_weights = dict(sorted(weights.items(), key=lambda x: x[1], reverse=True))
        
        # Create custom colormap
        cmap = plt.cm.viridis
        colors = cmap(np.linspace(0, 1, len(sorted_weights)))
        
        # Create pie chart
        wedges, texts, autotexts = plt.pie(
            sorted_weights.values(),
            labels=sorted_weights.keys(),
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            wedgeprops={'edgecolor': 'w', 'linewidth': 1}
        )
        
        # Style auto texts
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(9)
        
        # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.axis('equal')
        plt.title(title)
        
        return plt.gcf()
    
    def plot_efficient_frontier(self, portfolios, figsize=(10, 6)):
        """Plot the efficient frontier of portfolios"""
        plt.figure(figsize=figsize)
        
        # Extract data for plotting
        returns = [p['return'] for p in portfolios]
        volatilities = [p['volatility'] for p in portfolios]
        sharpe_ratios = [p['sharpe_ratio'] for p in portfolios]
        
        # Create scatter plot
        sc = plt.scatter(volatilities, returns, c=sharpe_ratios, cmap='viridis', 
                       s=30, alpha=0.7)
        
        # Add colorbar for Sharpe ratios
        cbar = plt.colorbar(sc)
        cbar.set_label('Sharpe Ratio')
        
        # Highlight max Sharpe ratio portfolio
        max_sharpe_idx = np.argmax(sharpe_ratios)
        plt.scatter(volatilities[max_sharpe_idx], returns[max_sharpe_idx], 
                   color='red', marker='*', s=200, label='Max Sharpe Ratio')
        
        # Highlight min volatility portfolio
        min_vol_idx = np.argmin(volatilities)
        plt.scatter(volatilities[min_vol_idx], returns[min_vol_idx], 
                   color='green', marker='*', s=200, label='Min Volatility')
        
        plt.title('Efficient Frontier')
        plt.xlabel('Volatility (Standard Deviation)')
        plt.ylabel('Expected Return')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        return plt.gcf()
    
    def plot_monte_carlo_simulation(self, simulation_results, title="Monte Carlo Simulation", figsize=(10, 6)):
        """Plot Monte Carlo simulation results"""
        plt.figure(figsize=figsize)
        
        # Plot the simulation results
        plt.plot(simulation_results.index, simulation_results['mean'] * 100, 
               label='Mean', color='blue', linewidth=2)
        plt.plot(simulation_results.index, simulation_results['median'] * 100, 
               label='Median', color='green', linewidth=2)
        
        # Plot confidence intervals
        plt.fill_between(simulation_results.index, 
                       simulation_results['percentile_5'] * 100, 
                       simulation_results['percentile_95'] * 100, 
                       color='gray', alpha=0.3, label='90% Confidence Interval')
        
        plt.fill_between(simulation_results.index, 
                       simulation_results['percentile_25'] * 100, 
                       simulation_results['percentile_75'] * 100, 
                       color='gray', alpha=0.5, label='50% Confidence Interval')
        
        # Plot min and max
        plt.plot(simulation_results.index, simulation_results['min'] * 100, 
               label='Min', color='red', alpha=0.5, linestyle='--')
        plt.plot(simulation_results.index, simulation_results['max'] * 100, 
               label='Max', color='green', alpha=0.5, linestyle='--')
        
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Cumulative Returns (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Format date axis if index is datetimes
        if isinstance(simulation_results.index, pd.DatetimeIndex):
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(plt.gca().get_xticklabels(), rotation=30, ha='right')
        
        return plt.gcf()
    
    def plot_correlation_matrix(self, returns, title="Asset Correlation Matrix", figsize=(10, 8)):
        """Plot correlation matrix of asset returns"""
        plt.figure(figsize=figsize)
        
        # Calculate correlation matrix
        corr_matrix = returns.corr()
        
        # Create custom colormap (blue to white to red)
        colors = ['blue', 'white', 'red']
        cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=100)
        
        # Create heatmap
        sns.heatmap(corr_matrix, annot=True, cmap=cmap, center=0,
                  linewidths=.5, fmt='.2f', vmin=-1, vmax=1)
        
        plt.title(title)
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_feature_importance(self, feature_importance, title="Feature Importance", figsize=(10, 8)):
        """Plot feature importance from ML model"""
        plt.figure(figsize=figsize)
        
        # Sort by importance
        feature_importance = feature_importance.sort_values('importance', ascending=True)
        
        # Plot horizontal bar chart
        plt.barh(feature_importance['feature'], feature_importance['importance'])
        
        plt.title(title)
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.grid(True, alpha=0.3)
        
        # Adjust layout to fit feature names
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_returns_distribution(self, returns, title="Returns Distribution", figsize=(10, 6)):
        """Plot distribution of returns"""
        plt.figure(figsize=figsize)
        
        # Create histogram with kde
        sns.histplot(returns * 100, kde=True, bins=50)
        
        # Add vertical lines for key statistics
        plt.axvline(returns.mean() * 100, color='r', linestyle='--', label=f'Mean: {returns.mean() * 100:.2f}%')
        plt.axvline(returns.median() * 100, color='g', linestyle='--', label=f'Median: {returns.median() * 100:.2f}%')
        plt.axvline(0, color='k', linestyle='-', alpha=0.3)
        
        # Add percentiles
        percentile_5 = np.percentile(returns, 5) * 100
        percentile_95 = np.percentile(returns, 95) * 100
        plt.axvline(percentile_5, color='orange', linestyle=':', label=f'5th Percentile: {percentile_5:.2f}%')
        plt.axvline(percentile_95, color='purple', linestyle=':', label=f'95th Percentile: {percentile_95:.2f}%')
        
        plt.title(title)
        plt.xlabel('Returns (%)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        return plt.gcf()