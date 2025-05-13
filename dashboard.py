# dashboard.py - Streamlit dashboard

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import io
import sys
import os

# Add parent directory to Python path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from financial_system import FinancialAnalysisSystem
from config import CONFIG

# Initialize the system
system = FinancialAnalysisSystem()

def main():
    """Main function for Streamlit dashboard"""
    st.set_page_config(page_title="Financial Analysis System", page_icon="📈", layout="wide")
    
    # Add CSS for styling
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #4285F4;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #34A853;
        }
        .metrics-container {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }
        .metric-box {
            background-color: #f1f3f4;
            border-radius: 5px;
            padding: 10px;
            flex: 1;
            min-width: 120px;
            text-align: center;
        }
        </style>
        """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">Financial Analysis Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("This dashboard allows you to analyze financial assets and build portfolios using advanced technical analysis and machine learning techniques.")
    
    # Create tabs
    tabs = st.tabs(["Asset Analysis", "Portfolio Optimization", "ML Predictions", "Risk Analysis"])
    
    # Asset Analysis Tab
    with tabs[0]:
        asset_analysis_tab()
    
    # Portfolio Optimization Tab
    with tabs[1]:
        portfolio_optimization_tab()
    
    # ML Predictions Tab
    with tabs[2]:
        ml_predictions_tab()
    
    # Risk Analysis Tab
    with tabs[3]:
        risk_analysis_tab()

def asset_analysis_tab():
    """Asset Analysis Tab"""
    st.markdown('<h2 class="sub-header">Asset Analysis</h2>', unsafe_allow_html=True)
    
    # Create sidebar for inputs
    st.sidebar.header("Asset Analysis Settings")
    
    # Asset type selection
    asset_type = st.sidebar.selectbox(
        "Select Asset Type",
        ["stock", "crypto"],
        index=0  # Default to stock
    )
    
    # Asset selection
    if asset_type == "stock":
        available_assets = CONFIG["stock_tickers"]
    else:
        available_assets = CONFIG["crypto_tickers"]
    
    selected_asset = st.sidebar.selectbox(
        "Select Asset",
        available_assets,
        index=0  # Default to first asset
    )
    
    # Time period selection
    time_period = st.sidebar.selectbox(
        "Select Time Period",
        ["3mo", "6mo", "1y", "2y", "5y"],
        index=2  # Default to 1y
    )
    
    # Include sentiment data
    include_sentiment = st.sidebar.checkbox("Include Sentiment Analysis", value=True)
    
    # Run analysis button
    if st.sidebar.button("Analyze Asset"):
        # Display loading spinner
        with st.spinner(f"Analyzing {selected_asset}..."):
            try:
                # Perform analysis
                backtest_result, metrics = system.analyze_asset(
                    asset_type, 
                    selected_asset, 
                    period=time_period,
                    include_sentiment=include_sentiment
                )
                
                # Display metrics
                st.markdown('<h3>Performance Metrics</h3>', unsafe_allow_html=True)
                
                # Create columns for metrics
                col1, col2, col3, col4 = st.columns(4)
                
                # Format metrics
                total_return_str = f"{metrics['total_return']:.2%}"
                annual_return_str = f"{metrics['annualized_return']:.2%}"
                max_drawdown_str = f"{metrics['max_drawdown']:.2%}"
                sharpe_ratio_str = f"{metrics['sharpe_ratio']:.2f}"
                
                # Add color based on value
                total_return_color = "green" if metrics['total_return'] > 0 else "red"
                annual_return_color = "green" if metrics['annualized_return'] > 0 else "red"
                
                # Display metrics with coloring
                col1.metric("Total Return", total_return_str)
                col2.metric("Annualized Return", annual_return_str)
                col3.metric("Max Drawdown", max_drawdown_str)
                col4.metric("Sharpe Ratio", sharpe_ratio_str)
                
                # Plot technical analysis
                st.markdown('<h3>Technical Analysis</h3>', unsafe_allow_html=True)
                fig1, fig2 = system.visualize_analysis(backtest_result, metrics, title=selected_asset)
                
                # Convert matplotlib figures to streamlit
                st.pyplot(fig1)
                
                # Plot backtest results
                st.markdown('<h3>Backtest Results</h3>', unsafe_allow_html=True)
                st.pyplot(fig2)
                
                # Display trade statistics if available
                if 'num_trades' in metrics:
                    st.markdown('<h3>Trade Statistics</h3>', unsafe_allow_html=True)
                    
                    # Create columns for trade metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    # Format metrics
                    num_trades_str = f"{metrics['num_trades']}"
                    win_rate_str = f"{metrics['win_rate']:.2%}" if 'win_rate' in metrics else "N/A"
                    avg_profit_str = f"{metrics['avg_profit_per_trade']:.2%}" if 'avg_profit_per_trade' in metrics else "N/A"
                    profit_factor_str = f"{metrics['profit_factor']:.2f}" if 'profit_factor' in metrics else "N/A"
                    
                    # Display metrics
                    col1.metric("Number of Trades", num_trades_str)
                    col2.metric("Win Rate", win_rate_str)
                    col3.metric("Avg. Profit per Trade", avg_profit_str)
                    col4.metric("Profit Factor", profit_factor_str)
                
                # Display current position
                st.markdown('<h3>Current Position</h3>', unsafe_allow_html=True)
                
                # Get final signal
                final_signal = backtest_result['final_signal'].iloc[-1]
                signal_str = "BUY" if final_signal == 1 else "SELL" if final_signal == -1 else "HOLD"
                signal_color = "green" if final_signal == 1 else "red" if final_signal == -1 else "gray"
                
                # Display signal
                st.markdown(f"<h4 style='color: {signal_color};'>Current Signal: {signal_str}</h4>", unsafe_allow_html=True)
                
                # Display recent price
                recent_price = backtest_result['close'].iloc[-1]
                st.markdown(f"Recent Price: ${recent_price:.2f}")
                
                # Display data table
                st.markdown('<h3>Recent Data</h3>', unsafe_allow_html=True)
                st.dataframe(backtest_result.tail(20))
                
            except Exception as e:
                st.error(f"Error analyzing {selected_asset}: {e}")

def portfolio_optimization_tab():
    """Portfolio Optimization Tab"""
    st.markdown('<h2 class="sub-header">Portfolio Optimization</h2>', unsafe_allow_html=True)
    
    # Create sidebar for inputs
    st.sidebar.header("Portfolio Settings")
    
    # Asset type selection
    asset_type = st.sidebar.selectbox(
        "Select Asset Type for Portfolio",
        ["stock", "crypto"],
        index=0,  # Default to stock
        key="portfolio_asset_type"
    )
    
    # Number of assets
    num_assets = st.sidebar.slider(
        "Number of Top Assets",
        min_value=2,
        max_value=10,
        value=5
    )
    
    # Time period selection
    time_period = st.sidebar.selectbox(
        "Analysis Period",
        ["3mo", "6mo", "1y"],
        index=0,  # Default to 3mo
        key="portfolio_time_period"
    )
    
    # Run optimization button
    if st.sidebar.button("Generate Portfolio"):
        # Display loading spinner
        with st.spinner("Optimizing portfolio..."):
            try:
                # Get portfolio recommendations
                portfolio_recommendations = system.get_portfolio_recommendations(
                    asset_type=asset_type,
                    top_n=num_assets,
                    period=time_period
                )
                
                # Display optimized portfolio
                st.markdown('<h3>Optimized Portfolio</h3>', unsafe_allow_html=True)
                
                # Create table of top assets
                top_assets_df = pd.DataFrame(portfolio_recommendations['top_assets'])
                
                # Format percentages
                top_assets_df['annualized_return'] = top_assets_df['annualized_return'].apply(lambda x: f"{x:.2%}")
                top_assets_df['max_drawdown'] = top_assets_df['max_drawdown'].apply(lambda x: f"{x:.2%}")
                
                # Map signals to text
                top_assets_df['signal'] = top_assets_df['signal'].map({1: "BUY", -1: "SELL", 0: "HOLD"})
                
                # Display table
                st.dataframe(top_assets_df)
                
                # Display portfolio metrics
                st.markdown('<h3>Portfolio Performance Metrics</h3>', unsafe_allow_html=True)
                
                # Create columns for metrics
                col1, col2, col3 = st.columns(3)
                
                # Format metrics
                port_return_str = f"{portfolio_recommendations['performance']['return']:.2%}"
                port_volatility_str = f"{portfolio_recommendations['performance']['volatility']:.2%}"
                port_sharpe_str = f"{portfolio_recommendations['performance']['sharpe_ratio']:.2f}"
                
                # Display metrics
                col1.metric("Expected Annual Return", port_return_str)
                col2.metric("Expected Volatility", port_volatility_str)
                col3.metric("Sharpe Ratio", port_sharpe_str)
                
                # Plot portfolio allocation
                st.markdown('<h3>Portfolio Allocation</h3>', unsafe_allow_html=True)
                allocation_chart, frontier_chart = system.visualize_portfolio(portfolio_recommendations)
                
                # Convert matplotlib figures to streamlit
                col1, col2 = st.columns(2)
                with col1:
                    st.pyplot(allocation_chart)
                
                # Plot efficient frontier
                with col2:
                    st.markdown('<h4>Efficient Frontier</h4>', unsafe_allow_html=True)
                    st.pyplot(frontier_chart)
                
                # Display weights
                st.markdown('<h3>Optimal Weights</h3>', unsafe_allow_html=True)
                
                # Convert weights to DataFrame
                weights_df = pd.DataFrame({
                    'Asset': list(portfolio_recommendations['optimized_weights'].keys()),
                    'Weight': list(portfolio_recommendations['optimized_weights'].values())
                })
                
                # Format percentages
                weights_df['Weight'] = weights_df['Weight'].apply(lambda x: f"{x:.2%}")
                
                # Display table
                st.dataframe(weights_df)
                
            except Exception as e:
                st.error(f"Error generating portfolio: {e}")

def ml_predictions_tab():
    """ML Predictions Tab"""
    st.markdown('<h2 class="sub-header">Machine Learning Predictions</h2>', unsafe_allow_html=True)
    
    # Create sidebar for inputs
    st.sidebar.header("ML Prediction Settings")
    
    # Asset type selection
    asset_type = st.sidebar.selectbox(
        "Select Asset Type",
        ["stock", "crypto"],
        index=0,  # Default to stock
        key="ml_asset_type"
    )
    
    # Asset selection
    if asset_type == "stock":
        available_assets = CONFIG["stock_tickers"]
    else:
        available_assets = CONFIG["crypto_tickers"]
    
    selected_asset = st.sidebar.selectbox(
        "Select Asset",
        available_assets,
        index=0,  # Default to first asset
        key="ml_asset"
    )
    
    # Time period selection
    time_period = st.sidebar.selectbox(
        "Training Period",
        ["1y", "2y", "5y"],
        index=0,  # Default to 1y
        key="ml_time_period"
    )
    
    # Prediction type
    prediction_type = st.sidebar.selectbox(
        "Prediction Type",
        ["direction", "close", "returns"],
        index=0  # Default to direction
    )
    
    # Prediction horizon
    prediction_horizon = st.sidebar.slider(
        "Prediction Horizon (days)",
        min_value=1,
        max_value=30,
        value=5
    )
    
    # Train model button
    if st.sidebar.button("Train Model"):
        # Display loading spinner
        with st.spinner(f"Training ML model for {selected_asset}..."):
            try:
                # Train model
                ml_results = system.train_ml_model(
                    asset_type,
                    selected_asset,
                    period=time_period,
                    prediction_horizon=prediction_horizon,
                    prediction_type=prediction_type
                )
                
                # Display training metrics
                st.markdown('<h3>Model Training Results</h3>', unsafe_allow_html=True)
                
                if prediction_type == 'direction':
                    # Classification metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    # Format metrics
                    accuracy_str = f"{ml_results['training_results']['accuracy']:.2%}"
                    precision_str = f"{ml_results['training_results']['precision']:.2%}"
                    recall_str = f"{ml_results['training_results']['recall']:.2%}"
                    f1_str = f"{ml_results['training_results']['f1_score']:.2f}"
                    
                    # Display metrics
                    col1.metric("Accuracy", accuracy_str)
                    col2.metric("Precision", precision_str)
                    col3.metric("Recall", recall_str)
                    col4.metric("F1 Score", f1_str)
                else:
                    # Regression metrics
                    col1, col2, col3 = st.columns(3)
                    
                    # Format metrics
                    mse_str = f"{ml_results['training_results']['mse']:.4f}"
                    rmse_str = f"{ml_results['training_results']['rmse']:.4f}"
                    r2_str = f"{ml_results['training_results']['r2_score']:.2f}"
                    
                    # Display metrics
                    col1.metric("MSE", mse_str)
                    col2.metric("RMSE", rmse_str)
                    col3.metric("R² Score", r2_str)
                
                # Plot feature importance
                st.markdown('<h3>Feature Importance</h3>', unsafe_allow_html=True)
                feature_chart = system.visualizer.plot_feature_importance(
                    ml_results['feature_importance'],
                    title=f"Feature Importance for {selected_asset} Prediction"
                )
                
                # Convert matplotlib figure to streamlit
                st.pyplot(feature_chart)
                
                # Display backtest results
                st.markdown('<h3>ML Strategy Backtest Results</h3>', unsafe_allow_html=True)
                
                # Create columns for metrics
                col1, col2, col3, col4 = st.columns(4)
                
                # Format metrics
                ml_return_str = f"{ml_results['backtest_metrics']['total_return']:.2%}"
                ml_annual_str = f"{ml_results['backtest_metrics']['annualized_return']:.2%}"
                ml_drawdown_str = f"{ml_results['backtest_metrics']['max_drawdown']:.2%}"
                ml_sharpe_str = f"{ml_results['backtest_metrics']['sharpe_ratio']:.2f}"
                
                # Display metrics
                col1.metric("Total Return", ml_return_str)
                col2.metric("Annualized Return", ml_annual_str)
                col3.metric("Max Drawdown", ml_drawdown_str)
                col4.metric("Sharpe Ratio", ml_sharpe_str)
                
                # Plot backtest results
                backtest_chart = system.visualizer.plot_backtest_results(
                    ml_results['backtest_results'],
                    ml_results['backtest_metrics'],
                    title=f"ML Strategy Backtest for {selected_asset}"
                )
                
                # Convert matplotlib figure to streamlit
                st.pyplot(backtest_chart)
                
                # Display cross-validation results
                st.markdown('<h3>Cross-Validation Results</h3>', unsafe_allow_html=True)
                
                # Convert cross-validation results to DataFrame
                cv_df = pd.DataFrame({k: [v] for k, v in ml_results['cross_validation'].items()})
                
                # Transpose for better display
                cv_df = cv_df.T.reset_index()
                cv_df.columns = ['Metric', 'Value']
                
                # Display table
                st.dataframe(cv_df)
                
                # Display predictions
                st.markdown('<h3>Recent Predictions</h3>', unsafe_allow_html=True)
                
                # Get recent predictions
                recent_predictions = ml_results['backtest_results'][['close', 'prediction', 'ml_signal']].tail(10)
                
                # Map signals to text
                recent_predictions['ml_signal'] = recent_predictions['ml_signal'].map({1: "BUY", -1: "SELL", 0: "HOLD"})
                
                # Display table
                st.dataframe(recent_predictions)
                
            except Exception as e:
                st.error(f"Error training model: {e}")

def risk_analysis_tab():
    """Risk Analysis Tab"""
    st.markdown('<h2 class="sub-header">Portfolio Risk Analysis</h2>', unsafe_allow_html=True)
    
    # Create sidebar for inputs
    st.sidebar.header("Risk Analysis Settings")
    
    # Multi-select for assets
    st.sidebar.markdown("**Select Assets for Portfolio**")
    
    # Stock selection
    stock_options = CONFIG["stock_tickers"]
    selected_stocks = st.sidebar.multiselect(
        "Stocks",
        stock_options,
        default=stock_options[:3]  # Default to first 3 stocks
    )
    
    # Crypto selection
    crypto_options = CONFIG["crypto_tickers"]
    selected_cryptos = st.sidebar.multiselect(
        "Cryptocurrencies",
        crypto_options,
        default=[]  # Default to no cryptos
    )
    
    # Combine selected assets
    portfolio_assets = selected_stocks + selected_cryptos
    
    # Time period selection
    time_period = st.sidebar.selectbox(
        "Analysis Period",
        ["6mo", "1y", "2y"],
        index=1,  # Default to 1y
        key="risk_time_period"
    )
    
    # Custom weights option
    use_custom_weights = st.sidebar.checkbox("Use Custom Weights")
    
    weights = {}
    if use_custom_weights and portfolio_assets:
        st.sidebar.markdown("**Custom Weights**")
        
        # Initialize with equal weights
        equal_weight = 1.0 / len(portfolio_assets)
        
        # Create sliders for weights
        for asset in portfolio_assets:
            weights[asset] = st.sidebar.slider(
                f"{asset} Weight",
                min_value=0.0,
                max_value=1.0,
                value=equal_weight,
                step=0.01,
                format="%.2f"
            )
        
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
    
    # Run analysis button
    if st.sidebar.button("Analyze Risk") and portfolio_assets:
        # Display loading spinner
        with st.spinner("Analyzing portfolio risk..."):
            try:
                # Analyze portfolio risk
                risk_results = system.analyze_portfolio_risk(
                    portfolio_assets,
                    weights=weights if use_custom_weights else None,
                    period=time_period
                )
                
                # Display portfolio risk metrics
                st.markdown('<h3>Portfolio Risk Metrics</h3>', unsafe_allow_html=True)
                
                # Get portfolio metrics
                portfolio_metrics = risk_results['portfolio_metrics'].iloc[0]
                
                # Create columns for metrics
                col1, col2, col3, col4 = st.columns(4)
                
                # Format metrics
                annual_return_str = f"{portfolio_metrics['annual_return']:.2%}"
                annual_vol_str = f"{portfolio_metrics['annual_volatility']:.2%}"
                sharpe_str = f"{portfolio_metrics['sharpe_ratio']:.2f}"
                sortino_str = f"{portfolio_metrics['sortino_ratio']:.2f}"
                
                # Display metrics
                col1.metric("Annual Return", annual_return_str)
                col2.metric("Annual Volatility", annual_vol_str)
                col3.metric("Sharpe Ratio", sharpe_str)
                col4.metric("Sortino Ratio", sortino_str)
                
                # Create columns for more metrics
                col1, col2, col3, col4 = st.columns(4)
                
                # Format metrics
                max_dd_str = f"{portfolio_metrics['max_drawdown']:.2%}"
                var_str = f"{portfolio_metrics['var_95']:.2%}"
                cvar_str = f"{portfolio_metrics['cvar_95']:.2%}"
                
                # Display metrics
                col1.metric("Max Drawdown", max_dd_str)
                col2.metric("VaR (95%)", var_str)
                col3.metric("CVaR (95%)", cvar_str)
                
                # Plot correlation matrix
                st.markdown('<h3>Asset Correlation Matrix</h3>', unsafe_allow_html=True)
                corr_chart = system.visualizer.plot_correlation_matrix(
                    risk_results['correlation_matrix'],
                    title="Asset Correlation Matrix"
                )
                
                # Convert matplotlib figure to streamlit
                st.pyplot(corr_chart)
                
                # Plot Monte Carlo simulation
                st.markdown('<h3>Monte Carlo Simulation</h3>', unsafe_allow_html=True)
                mc_chart = system.visualizer.plot_monte_carlo_simulation(
                    risk_results['simulation_results'],
                    title="Portfolio Monte Carlo Simulation (1000 scenarios)"
                )
                
                # Convert matplotlib figure to streamlit
                st.pyplot(mc_chart)
                
                # Display asset risk metrics
                st.markdown('<h3>Individual Asset Risk Metrics</h3>', unsafe_allow_html=True)
                
                # Format asset metrics table
                asset_metrics = risk_results['asset_metrics'].copy()
                
                # Convert to percentages
                for col in ['annual_return', 'annual_volatility', 'max_drawdown', 'var_95', 'cvar_95']:
                    asset_metrics[col] = asset_metrics[col].apply(lambda x: f"{x:.2%}")
                
                # Display table
                st.dataframe(asset_metrics)
                
            except Exception as e:
                st.error(f"Error analyzing portfolio risk: {e}")
    elif not portfolio_assets:
        st.warning("Please select at least one asset for the portfolio.")

if __name__ == "__main__":
    main()