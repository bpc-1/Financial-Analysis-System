# backtester.py - Strategy backtesting

import pandas as pd
import numpy as np
import logging
from utils import calculate_sharpe_ratio, calculate_drawdown

logger = logging.getLogger(__name__)

class Backtester:
    """Class for backtesting trading strategies"""
    
    def __init__(self, initial_capital=10000):
        """Initialize with starting capital"""
        self.initial_capital = initial_capital
    
    def backtest_strategy(self, df, commission=0.001, slippage=0.0005):
        """Backtest a trading strategy based on 'final_signal' column
        
        Args:
            df: DataFrame with price data and 'final_signal' column
            commission: Commission rate as a decimal (e.g., 0.001 for 0.1%)
            slippage: Slippage rate as a decimal (e.g., 0.0005 for 0.05%)
        
        Returns:
            DataFrame with backtest results and performance metrics
        """
        # Make a copy of the DataFrame
        result = df.copy()
        
        # Verify we have required columns
        if 'close' not in result.columns:
            raise ValueError("DataFrame must have a 'close' column for backtesting")
        
        if 'final_signal' not in result.columns:
            raise ValueError("DataFrame must have a 'final_signal' column for backtesting")
        
        # Initialize columns with explicit float types
        result['position'] = 0.0
        result['cash'] = float(self.initial_capital)
        result['holdings'] = 0.0
        result['portfolio_value'] = float(self.initial_capital)
        result['returns'] = 0.0
        result['trade_type'] = ''  # 'buy', 'sell', or ''
        result['trade_price'] = 0.0
        result['trade_shares'] = 0.0
        
        # Iterate through the DataFrame
        position = 0.0
        cash = float(self.initial_capital)
        for i in range(1, len(result)):
            prev_row = result.iloc[i-1]
            curr_row = result.iloc[i]
            
            # Apply slippage to trade prices
            entry_price = curr_row['close'] * (1 + slippage)  # Buy at higher price
            exit_price = curr_row['close'] * (1 - slippage)   # Sell at lower price
            
            trade_type = ''
            trade_price = 0.0
            trade_shares = 0.0
            
            # Update position based on signal
            if curr_row['final_signal'] == 1 and position == 0:
                # Buy signal and no position -> Buy
                trade_shares = cash / entry_price
                trade_shares = trade_shares * (1 - commission)  # Account for commission
                position = trade_shares
                cash = 0
                trade_type = 'buy'
                trade_price = entry_price
            elif curr_row['final_signal'] == -1 and position > 0:
                # Sell signal and has position -> Sell
                trade_shares = position
                cash = position * exit_price
                cash = cash * (1 - commission)  # Account for commission
                position = 0
                trade_type = 'sell'
                trade_price = exit_price
            
            # Update result DataFrame
            result.iloc[i, result.columns.get_loc('position')] = position
            result.iloc[i, result.columns.get_loc('cash')] = cash
            result.iloc[i, result.columns.get_loc('holdings')] = position * curr_row['close']
            result.iloc[i, result.columns.get_loc('portfolio_value')] = cash + (position * curr_row['close'])
            result.iloc[i, result.columns.get_loc('trade_type')] = trade_type
            result.iloc[i, result.columns.get_loc('trade_price')] = trade_price
            result.iloc[i, result.columns.get_loc('trade_shares')] = trade_shares
            
            # Calculate returns
            prev_portfolio_value = prev_row['portfolio_value']
            curr_portfolio_value = result.iloc[i, result.columns.get_loc('portfolio_value')]
            result.iloc[i, result.columns.get_loc('returns')] = (curr_portfolio_value / prev_portfolio_value) - 1
        
        # Calculate cumulative returns
        result['cumulative_returns'] = (1 + result['returns']).cumprod() - 1
        
        # Calculate drawdown
        result['drawdown'] = calculate_drawdown(result['portfolio_value'])
        
        # Calculate performance metrics
        total_return = result['portfolio_value'].iloc[-1] / self.initial_capital - 1
        
        # Calculate annualized return
        days = (result.index[-1] - result.index[0]).days
        years = max(days / 365, 0.01)  # Avoid division by zero
        annualized_return = (1 + total_return) ** (1 / years) - 1
        
        # Calculate maximum drawdown
        max_drawdown = result['drawdown'].max()
        
        # Calculate average drawdown
        avg_drawdown = result['drawdown'].mean()
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0 for simplicity)
        daily_returns = result['returns'].dropna()
        sharpe_ratio = calculate_sharpe_ratio(daily_returns)
        
        # Calculate number of trades and win rate
        buy_trades = result[result['trade_type'] == 'buy']
        sell_trades = result[result['trade_type'] == 'sell']
        num_trades = min(len(buy_trades), len(sell_trades))  # Paired trades
        
        # Match buys and sells to calculate profit per trade
        profitable_trades = 0
        profit_per_trade = []
        
        if num_trades > 0:
            buy_indices = buy_trades.index.tolist()
            sell_indices = sell_trades.index.tolist()
            
            # Ensure we have paired trades
            paired_trades = min(len(buy_indices), len(sell_indices))
            
            for i in range(paired_trades):
                buy_price = result.loc[buy_indices[i], 'trade_price']
                sell_price = result.loc[sell_indices[i], 'trade_price']
                
                trade_profit = (sell_price / buy_price) - 1 - 2 * commission
                profit_per_trade.append(trade_profit)
                
                if trade_profit > 0:
                    profitable_trades += 1
            
            win_rate = profitable_trades / paired_trades if paired_trades > 0 else 0
            avg_profit_per_trade = np.mean(profit_per_trade) if profit_per_trade else 0
            avg_win = np.mean([p for p in profit_per_trade if p > 0]) if [p for p in profit_per_trade if p > 0] else 0
            avg_loss = np.mean([p for p in profit_per_trade if p <= 0]) if [p for p in profit_per_trade if p <= 0] else 0
            profit_factor = -avg_win / avg_loss if avg_loss != 0 else float('inf')
        else:
            win_rate = 0
            avg_profit_per_trade = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_profit_per_trade': avg_profit_per_trade,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss
        }
        
        return result, metrics
    
    def backtest_portfolio(self, df_dict, weights, commission=0.001, slippage=0.0005, rebalance_frequency='M'):
        """
        Backtest a portfolio of assets with fixed weights and periodic rebalancing
        
        Args:
            df_dict: Dictionary of DataFrames with price data and 'final_signal' column for each asset
            weights: Dictionary of asset weights
            commission: Commission rate as a decimal
            slippage: Slippage rate as a decimal
            rebalance_frequency: Pandas frequency string for rebalancing ('D' for daily, 'W' for weekly, 'M' for monthly)
        
        Returns:
            DataFrame with portfolio backtest results and performance metrics
        """
        # Verify we have data for all assets in weights
        for asset in weights:
            if asset not in df_dict:
                raise ValueError(f"Missing data for asset: {asset}")
        
        # Extract close prices and final signals for all assets
        close_prices = {}
        signals = {}
        
        for asset, df in df_dict.items():
            if 'close' not in df.columns:
                raise ValueError(f"Missing 'close' column for asset: {asset}")
            
            close_prices[asset] = df['close']
            
            if 'final_signal' in df.columns:
                signals[asset] = df['final_signal']
        
        # Create a DataFrame with close prices for all assets
        prices_df = pd.DataFrame(close_prices)
        
        # Create a DataFrame with signals for all assets
        signals_df = pd.DataFrame(signals)
        
        # Initialize portfolio DataFrame
        portfolio = pd.DataFrame(index=prices_df.index)
        portfolio['cash'] = float(self.initial_capital)
        
        # Add columns for shares held in each asset
        for asset in weights:
            portfolio[f'{asset}_shares'] = 0.0
        
        # Add columns for value of each asset
        for asset in weights:
            portfolio[f'{asset}_value'] = 0.0
        
        portfolio['portfolio_value'] = float(self.initial_capital)
        portfolio['returns'] = 0.0
        portfolio['rebalance'] = False
        
        # Determine rebalance dates
        rebalance_dates = prices_df.resample(rebalance_frequency).first().index
        portfolio.loc[rebalance_dates, 'rebalance'] = True
        
        # Initial investment - allocate according to weights
        for asset, weight in weights.items():
            if np.isnan(prices_df[asset].iloc[0]):
                continue
                
            # Calculate shares to buy
            investment = self.initial_capital * weight
            shares = investment / prices_df[asset].iloc[0]
            shares = shares * (1 - commission)  # Account for commission
            
            # Update portfolio
            portfolio.loc[prices_df.index[0], f'{asset}_shares'] = shares
            portfolio.loc[prices_df.index[0], f'{asset}_value'] = shares * prices_df[asset].iloc[0]
            portfolio.loc[prices_df.index[0], 'cash'] -= investment
        
        # Initial portfolio value
        portfolio.loc[prices_df.index[0], 'portfolio_value'] = \
            portfolio.loc[prices_df.index[0], 'cash'] + \
            sum(portfolio.loc[prices_df.index[0], f'{asset}_value'] for asset in weights)
        
        # Iterate through each date
        for i in range(1, len(portfolio)):
            current_date = portfolio.index[i]
            prev_date = portfolio.index[i-1]
            
            # Carry forward shares from previous day
            for asset in weights:
                portfolio.loc[current_date, f'{asset}_shares'] = portfolio.loc[prev_date, f'{asset}_shares']
            
            # Update asset values based on current prices
            for asset in weights:
                if np.isnan(prices_df.loc[current_date, asset]):
                    portfolio.loc[current_date, f'{asset}_value'] = portfolio.loc[prev_date, f'{asset}_value']
                else:
                    portfolio.loc[current_date, f'{asset}_value'] = \
                        portfolio.loc[current_date, f'{asset}_shares'] * prices_df.loc[current_date, asset]
            
            # Initialize cash with previous day's value
            portfolio.loc[current_date, 'cash'] = portfolio.loc[prev_date, 'cash']
            
            # Check if we need to rebalance
            if portfolio.loc[current_date, 'rebalance']:
                # Calculate total portfolio value
                total_value = portfolio.loc[current_date, 'cash'] + \
                    sum(portfolio.loc[current_date, f'{asset}_value'] for asset in weights)
                
                # Calculate target value for each asset
                target_values = {asset: total_value * weight for asset, weight in weights.items()}
                
                # Sell assets above target weight
                for asset in weights:
                    if np.isnan(prices_df.loc[current_date, asset]):
                        continue
                        
                    current_value = portfolio.loc[current_date, f'{asset}_value']
                    if current_value > target_values[asset]:
                        # Calculate shares to sell
                        excess_value = current_value - target_values[asset]
                        shares_to_sell = excess_value / prices_df.loc[current_date, asset]
                        
                        # Adjust for commission
                        shares_to_sell = shares_to_sell / (1 + commission)
                        
                        # Update shares and values
                        portfolio.loc[current_date, f'{asset}_shares'] -= shares_to_sell
                        portfolio.loc[current_date, f'{asset}_value'] = \
                            portfolio.loc[current_date, f'{asset}_shares'] * prices_df.loc[current_date, asset]
                        
                        # Update cash
                        sell_proceeds = shares_to_sell * prices_df.loc[current_date, asset] * (1 - commission)
                        portfolio.loc[current_date, 'cash'] += sell_proceeds
                
                # Buy assets below target weight
                for asset in weights:
                    if np.isnan(prices_df.loc[current_date, asset]):
                        continue
                        
                    current_value = portfolio.loc[current_date, f'{asset}_value']
                    if current_value < target_values[asset]:
                        # Calculate shares to buy
                        shortfall = target_values[asset] - current_value
                        shares_to_buy = shortfall / prices_df.loc[current_date, asset]
                        
                        # Check if we have enough cash
                        cash_needed = shares_to_buy * prices_df.loc[current_date, asset] * (1 + commission)
                        if cash_needed > portfolio.loc[current_date, 'cash']:
                            # Adjust shares based on available cash
                            shares_to_buy = portfolio.loc[current_date, 'cash'] / \
                                (prices_df.loc[current_date, asset] * (1 + commission))
                        
                        # Update shares and values
                        portfolio.loc[current_date, f'{asset}_shares'] += shares_to_buy
                        portfolio.loc[current_date, f'{asset}_value'] = \
                            portfolio.loc[current_date, f'{asset}_shares'] * prices_df.loc[current_date, asset]
                        
                        # Update cash
                        buy_cost = shares_to_buy * prices_df.loc[current_date, asset] * (1 + commission)
                        portfolio.loc[current_date, 'cash'] -= buy_cost
            
            # Update portfolio value
            portfolio.loc[current_date, 'portfolio_value'] = \
                portfolio.loc[current_date, 'cash'] + \
                sum(portfolio.loc[current_date, f'{asset}_value'] for asset in weights)
            
            # Calculate returns
            portfolio.loc[current_date, 'returns'] = \
                (portfolio.loc[current_date, 'portfolio_value'] / portfolio.loc[prev_date, 'portfolio_value']) - 1
        
        # Calculate cumulative returns
        portfolio['cumulative_returns'] = (1 + portfolio['returns']).cumprod() - 1
        
        # Calculate drawdown
        portfolio['drawdown'] = calculate_drawdown(portfolio['portfolio_value'])
        
        # Calculate performance metrics
        total_return = portfolio['portfolio_value'].iloc[-1] / self.initial_capital - 1
        
        # Calculate annualized return
        days = (portfolio.index[-1] - portfolio.index[0]).days
        years = max(days / 365, 0.01)  # Avoid division by zero
        annualized_return = (1 + total_return) ** (1 / years) - 1
        
        # Calculate maximum drawdown
        max_drawdown = portfolio['drawdown'].max()
        
        # Calculate Sharpe ratio
        daily_returns = portfolio['returns'].dropna()
        sharpe_ratio = calculate_sharpe_ratio(daily_returns)
        
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'rebalance_count': portfolio['rebalance'].sum()
        }
        
        return portfolio, metrics
    
    def monte_carlo_simulation(self, df, num_simulations=1000):
        """
        Run Monte Carlo simulations to estimate the range of possible outcomes
        
        Args:
            df: DataFrame with backtest results
            num_simulations: Number of Monte Carlo simulations to run
        
        Returns:
            DataFrame with simulation results
        """
        # Extract daily returns from backtest
        returns = df['returns'].dropna()
        
        # Initialize simulation results
        simulation_results = pd.DataFrame()
        
        # Run simulations
        for i in range(num_simulations):
            # Resample returns with replacement
            sampled_returns = np.random.choice(returns, size=len(returns), replace=True)
            
            # Calculate cumulative returns
            cumulative_returns = (1 + sampled_returns).cumprod() - 1
            
            # Add to simulation results
            simulation_results[f'sim_{i}'] = cumulative_returns
        
        # Calculate statistics
        simulation_stats = pd.DataFrame()
        simulation_stats['mean'] = simulation_results.mean(axis=1)
        simulation_stats['median'] = simulation_results.median(axis=1)
        simulation_stats['min'] = simulation_results.min(axis=1)
        simulation_stats['max'] = simulation_results.max(axis=1)
        simulation_stats['percentile_5'] = simulation_results.quantile(0.05, axis=1)
        simulation_stats['percentile_25'] = simulation_results.quantile(0.25, axis=1)
        simulation_stats['percentile_75'] = simulation_results.quantile(0.75, axis=1)
        simulation_stats['percentile_95'] = simulation_results.quantile(0.95, axis=1)
        
        return simulation_stats