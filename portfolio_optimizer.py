# portfolio_optimizer.py - Portfolio optimization

import pandas as pd
import numpy as np
import logging
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    """Class for portfolio optimization and management"""
    
    def __init__(self, risk_free_rate=0.02):
        """Initialize with risk-free rate"""
        self.risk_free_rate = risk_free_rate
    
    def calculate_returns(self, price_data):
        """Calculate daily returns from price data"""
        return price_data.pct_change().dropna()
    
    def calculate_portfolio_performance(self, returns, weights):
        """Calculate expected return and risk for a portfolio"""
        weights = np.array(weights)
        
        # Expected portfolio return
        portfolio_return = np.sum(returns.mean() * weights) * 252  # Annualized
        
        # Expected portfolio volatility (risk)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        
        # Sharpe ratio
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        
        return {
            'return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio
        }
    
    def _negative_sharpe(self, weights, returns):
        """
        Returns the negative Sharpe Ratio for minimization
        """
        weights = np.array(weights)
        portfolio_return = np.sum(returns.mean() * weights) * 252
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        
        # Avoid division by zero
        if portfolio_volatility == 0:
            return 0
        
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        return -sharpe_ratio
    
    def optimize_weights(self, returns, target_return=None, target_risk=None, method='max_sharpe'):
        """
        Optimize portfolio weights based on different objectives
        
        Parameters:
        - returns: DataFrame of asset returns
        - target_return: Target portfolio return (for min_risk method)
        - target_risk: Target portfolio risk (for max_return method)
        - method: Optimization method ('max_sharpe', 'min_risk', 'max_return', 'equal_weight', 'risk_parity')
        
        Returns:
        - Dictionary with optimized weights and performance metrics
        """
        n = len(returns.columns)
        assets = returns.columns.tolist()
        
        if method == 'equal_weight':
            # Equal weight portfolio
            weights = np.ones(n) / n
            performance = self.calculate_portfolio_performance(returns, weights)
            
            return {
                'weights': dict(zip(assets, weights)),
                'performance': performance
            }
        
        elif method == 'max_sharpe':
            # Maximize Sharpe ratio
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Sum of weights = 1
            bounds = tuple((0, 1) for _ in range(n))  # 0 <= weight <= 1
            initial_weights = np.ones(n) / n  # Start with equal weights
            
            result = minimize(
                self._negative_sharpe,
                initial_weights,
                args=(returns,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            weights = result['x']
            performance = self.calculate_portfolio_performance(returns, weights)
            
            return {
                'weights': dict(zip(assets, weights)),
                'performance': performance
            }
        
        elif method == 'min_risk':
            # Minimize portfolio risk, optionally with a target return
            def portfolio_volatility(weights, returns):
                weights = np.array(weights)
                return np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
            
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Sum of weights = 1
            
            if target_return is not None:
                # Add target return constraint
                def portfolio_return(weights, returns):
                    weights = np.array(weights)
                    return np.sum(returns.mean() * weights) * 252
                
                constraints.append({
                    'type': 'eq',
                    'fun': lambda x: portfolio_return(x, returns) - target_return
                })
            
            bounds = tuple((0, 1) for _ in range(n))  # 0 <= weight <= 1
            initial_weights = np.ones(n) / n  # Start with equal weights
            
            result = minimize(
                portfolio_volatility,
                initial_weights,
                args=(returns,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            weights = result['x']
            performance = self.calculate_portfolio_performance(returns, weights)
            
            return {
                'weights': dict(zip(assets, weights)),
                'performance': performance
            }
        
        elif method == 'max_return':
            # Maximize portfolio return, optionally with a target risk
            def negative_portfolio_return(weights, returns):
                weights = np.array(weights)
                return -np.sum(returns.mean() * weights) * 252
            
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Sum of weights = 1
            
            if target_risk is not None:
                # Add target risk constraint
                def portfolio_volatility(weights, returns):
                    weights = np.array(weights)
                    return np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
                
                constraints.append({
                    'type': 'eq',
                    'fun': lambda x: portfolio_volatility(x, returns) - target_risk
                })
            
            bounds = tuple((0, 1) for _ in range(n))  # 0 <= weight <= 1
            initial_weights = np.ones(n) / n  # Start with equal weights
            
            result = minimize(
                negative_portfolio_return,
                initial_weights,
                args=(returns,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            weights = result['x']
            performance = self.calculate_portfolio_performance(returns, weights)
            
            return {
                'weights': dict(zip(assets, weights)),
                'performance': performance
            }
        
        elif method == 'risk_parity':
            # Risk parity portfolio (equal risk contribution)
            def risk_contribution(weights, returns):
                weights = np.array(weights)
                cov = returns.cov().values * 252
                portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
                
                # Marginal risk contribution
                marginal_risk = np.dot(cov, weights)
                
                # Risk contribution of each asset
                risk_contribution = weights * marginal_risk
                
                # Normalize by portfolio volatility
                if portfolio_volatility > 0:
                    risk_contribution = risk_contribution / portfolio_volatility
                else:
                    risk_contribution = np.ones_like(risk_contribution) / len(weights)
                
                return risk_contribution
            
            def risk_parity_objective(weights, returns):
                weights = np.array(weights)
                risk_contrib = risk_contribution(weights, returns)
                target_risk_contrib = 1 / len(weights)  # Equal risk contribution
                return np.sum((risk_contrib - target_risk_contrib)**2)
            
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Sum of weights = 1
            bounds = tuple((0.01, 1) for _ in range(n))  # 0.01 <= weight <= 1 (avoid 0 weights)
            initial_weights = np.ones(n) / n  # Start with equal weights
            
            result = minimize(
                risk_parity_objective,
                initial_weights,
                args=(returns,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            weights = result['x']
            performance = self.calculate_portfolio_performance(returns, weights)
            
            return {
                'weights': dict(zip(assets, weights)),
                'performance': performance
            }
        
        else:
            raise ValueError(f"Unknown optimization method: {method}")
    
    def generate_efficient_frontier(self, returns, num_portfolios=100):
        """Generate the efficient frontier of portfolios"""
        n = len(returns.columns)
        assets = returns.columns.tolist()
        
        # Find the minimum risk portfolio
        min_risk_portfolio = self.optimize_weights(returns, method='min_risk')
        min_risk = min_risk_portfolio['performance']['volatility']
        
        # Find a high return portfolio (not necessarily on the efficient frontier)
        # We'll use this to establish a range of returns
        max_weights = np.zeros(n)
        max_weights[returns.mean().argmax()] = 1  # Allocate 100% to the highest return asset
        max_return_performance = self.calculate_portfolio_performance(returns, max_weights)
        max_return = max_return_performance['return']
        
        # Generate target returns equally spaced from min risk portfolio return to max return
        target_returns = np.linspace(
            min_risk_portfolio['performance']['return'],
            max_return,
            num_portfolios
        )
        
        # For each target return, find the minimum risk portfolio
        portfolios = []
        for target_return in target_returns:
            try:
                portfolio = self.optimize_weights(returns, target_return=target_return, method='min_risk')
                portfolios.append({
                    'return': portfolio['performance']['return'],
                    'volatility': portfolio['performance']['volatility'],
                    'sharpe_ratio': portfolio['performance']['sharpe_ratio'],
                    'weights': portfolio['weights']
                })
            except:
                # Optimization might fail for some target returns
                pass
        
        # Also include the max Sharpe ratio portfolio
        max_sharpe_portfolio = self.optimize_weights(returns, method='max_sharpe')
        portfolios.append({
            'return': max_sharpe_portfolio['performance']['return'],
            'volatility': max_sharpe_portfolio['performance']['volatility'],
            'sharpe_ratio': max_sharpe_portfolio['performance']['sharpe_ratio'],
            'weights': max_sharpe_portfolio['weights'],
            'max_sharpe': True
        })
        
        return portfolios
    
    def calculate_risk_metrics(self, returns):
        """Calculate risk metrics for a portfolio or individual assets"""
        # Annualized return
        annual_return = returns.mean() * 252
        
        # Annualized volatility
        annual_volatility = returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        sharpe_ratio = (annual_return - self.risk_free_rate) / annual_volatility
        
        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        drawdown = 1 - cumulative_returns / cumulative_returns.cummax()
        max_drawdown = drawdown.max()
        
        # Sortino ratio (downside risk only)
        negative_returns = returns[returns < 0]
        downside_deviation = negative_returns.std() * np.sqrt(252)
        sortino_ratio = np.where(
            downside_deviation == 0,
            0,  # Avoid division by zero
            (annual_return - self.risk_free_rate) / downside_deviation
        )
        
        # Value at Risk (VaR)
        var_95 = np.percentile(returns, 5)
        
        # Conditional Value at Risk (CVaR) or Expected Shortfall
        cvar_95 = returns[returns <= var_95].mean()
        
        return pd.DataFrame({
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'cvar_95': cvar_95
        })
    
    def calculate_kelly_criterion(self, returns):
        """Calculate Kelly Criterion for optimal position sizing"""
        win_prob = len(returns[returns > 0]) / len(returns)
        win_avg = returns[returns > 0].mean()
        loss_avg = abs(returns[returns < 0].mean())
        
        # Kelly fraction
        kelly_fraction = win_prob - ((1 - win_prob) / (win_avg / loss_avg))
        
        # Often traders use a fraction of Kelly (half-Kelly) to reduce risk
        half_kelly = kelly_fraction / 2
        
        return {
            'kelly_fraction': kelly_fraction,
            'half_kelly': half_kelly,
            'win_probability': win_prob,
            'win_loss_ratio': win_avg / loss_avg
        }
    
    def calculate_dynamic_asset_allocation(self, returns, signal_df, risk_tolerance=0.5):
        """
        Calculate dynamic asset allocation based on signals and risk tolerance
        
        Parameters:
        - returns: DataFrame of asset returns
        - signal_df: DataFrame with 'final_signal' column (-1, 0, 1) for each asset
        - risk_tolerance: 0 to 1, higher values mean more aggressive allocation
        
        Returns:
        - Dictionary with asset allocations
        """
        assets = returns.columns.tolist()
        
        # Get the last signal for each asset
        last_signals = {}
        for asset in assets:
            if asset in signal_df.columns and 'final_signal' in signal_df.columns:
                # Filter signal_df to include only rows for this asset
                asset_signals = signal_df['final_signal'].iloc[-1]
                last_signals[asset] = asset_signals
            else:
                last_signals[asset] = 0
        
        # Calculate risk metrics
        risk_metrics = self.calculate_risk_metrics(returns)
        
        # Base weights (equal weight)
        base_weights = {asset: 1/len(assets) for asset in assets}
        
        # Adjust weights based on signals and risk metrics
        adjusted_weights = {}
        for asset in assets:
            signal = last_signals.get(asset, 0)
            sharpe = risk_metrics.loc[asset, 'sharpe_ratio'] if asset in risk_metrics.index else 0
            max_dd = risk_metrics.loc[asset, 'max_drawdown'] if asset in risk_metrics.index else 1
            
            # Scale for signal: -1 (sell) to 1 (buy)
            signal_factor = 1 + (signal * risk_tolerance)
            
            # Scale for Sharpe ratio
            sharpe_factor = 1 + (sharpe * risk_tolerance * 0.5)
            
            # Scale for max drawdown (inversely related)
            dd_factor = 1 - (max_dd * risk_tolerance * 0.5)
            
            # Combine factors
            combined_factor = signal_factor * sharpe_factor * dd_factor
            
            # Apply to base weight
            adjusted_weights[asset] = base_weights[asset] * combined_factor
        
        # Normalize weights to sum to 1
        total_weight = sum(adjusted_weights.values())
        normalized_weights = {asset: weight / total_weight for asset, weight in adjusted_weights.items()}
        
        return normalized_weights