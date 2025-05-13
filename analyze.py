# analyze.py - Example usage script

import argparse
import logging
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from financial_system import FinancialAnalysisSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("analysis_run.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main function for the analysis script"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Financial Analysis System')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Analyze asset command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a single asset')
    analyze_parser.add_argument('--type', '-t', choices=['stock', 'crypto'], required=True, help='Asset type')
    analyze_parser.add_argument('--symbol', '-s', required=True, help='Asset symbol/ticker')
    analyze_parser.add_argument('--period', '-p', default='1y', choices=['1mo', '3mo', '6mo', '1y', '2y', '5y'], help='Analysis period')
    analyze_parser.add_argument('--sentiment', action='store_true', help='Include sentiment analysis')
    analyze_parser.add_argument('--output', '-o', help='Output file for charts (without extension)')
    
    # Portfolio command
    portfolio_parser = subparsers.add_parser('portfolio', help='Create optimized portfolio')
    portfolio_parser.add_argument('--type', '-t', choices=['stock', 'crypto'], required=True, help='Asset type')
    portfolio_parser.add_argument('--count', '-c', type=int, default=5, help='Number of assets in portfolio')
    portfolio_parser.add_argument('--period', '-p', default='3mo', choices=['1mo', '3mo', '6mo', '1y'], help='Analysis period')
    portfolio_parser.add_argument('--output', '-o', help='Output file for charts (without extension)')
    
    # ML prediction command
    ml_parser = subparsers.add_parser('ml', help='Train ML model for prediction')
    ml_parser.add_argument('--type', '-t', choices=['stock', 'crypto'], required=True, help='Asset type')
    ml_parser.add_argument('--symbol', '-s', required=True, help='Asset symbol/ticker')
    ml_parser.add_argument('--predict', '-p', choices=['direction', 'close', 'returns'], default='direction', help='Prediction type')
    ml_parser.add_argument('--horizon', type=int, default=5, help='Prediction horizon in days')
    ml_parser.add_argument('--period', default='1y', choices=['1y', '2y', '5y'], help='Training period')
    ml_parser.add_argument('--output', '-o', help='Output file for charts (without extension)')
    
    # Risk analysis command
    risk_parser = subparsers.add_parser('risk', help='Analyze portfolio risk')
    risk_parser.add_argument('--symbols', '-s', required=True, nargs='+', help='Space-separated list of asset symbols')
    risk_parser.add_argument('--period', '-p', default='1y', choices=['3mo', '6mo', '1y', '2y'], help='Analysis period')
    risk_parser.add_argument('--output', '-o', help='Output file for charts (without extension)')
    
    args = parser.parse_args()
    
    # Create the system
    system = FinancialAnalysisSystem()
    
    # Execute command
    if args.command == 'analyze':
        run_analyze_command(system, args)
    elif args.command == 'portfolio':
        run_portfolio_command(system, args)
    elif args.command == 'ml':
        run_ml_command(system, args)
    elif args.command == 'risk':
        run_risk_command(system, args)
    else:
        parser.print_help()

def run_analyze_command(system, args):
    """Run analyze asset command"""
    logger.info(f"Analyzing {args.type}: {args.symbol} for period {args.period}")
    
    try:
        # Analyze asset
        backtest_result, metrics = system.analyze_asset(
            args.type,
            args.symbol,
            period=args.period,
            include_sentiment=args.sentiment
        )
        
        # Print metrics
        print("\n=== Performance Metrics ===")
        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"Annualized Return: {metrics['annualized_return']:.2%}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        
        if 'num_trades' in metrics:
            print("\n=== Trade Statistics ===")
            print(f"Number of Trades: {metrics['num_trades']}")
            print(f"Win Rate: {metrics['win_rate']:.2%}")
            print(f"Average Profit per Trade: {metrics['avg_profit_per_trade']:.2%}")
            print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        
        # Get final signal
        final_signal = backtest_result['final_signal'].iloc[-1]
        signal_str = "BUY" if final_signal == 1 else "SELL" if final_signal == -1 else "HOLD"
        
        print(f"\nCurrent Signal: {signal_str}")
        print(f"Recent Price: ${backtest_result['close'].iloc[-1]:.2f}")
        
        # Create visualizations
        price_chart, backtest_chart = system.visualize_analysis(backtest_result, metrics, title=args.symbol)
        
        # Show plots
        plt.show()
        
        # Save plots if output file specified
        if args.output:
            price_chart.savefig(f"{args.output}_technical.png")
            backtest_chart.savefig(f"{args.output}_backtest.png")
            logger.info(f"Charts saved to {args.output}_technical.png and {args.output}_backtest.png")
    
    except Exception as e:
        logger.error(f"Error analyzing {args.symbol}: {e}")
        raise

def run_portfolio_command(system, args):
    """Run portfolio optimization command"""
    logger.info(f"Creating optimized portfolio of {args.count} {args.type}s for period {args.period}")
    
    try:
        # Get portfolio recommendations
        portfolio = system.get_portfolio_recommendations(
            asset_type=args.type,
            top_n=args.count,
            period=args.period
        )
        
        # Print top assets
        print("\n=== Top Assets ===")
        for i, asset in enumerate(portfolio['top_assets']):
            signal_str = "BUY" if asset['signal'] == 1 else "SELL" if asset['signal'] == -1 else "HOLD"
            print(f"{i+1}. {asset['ticker']} - Return: {asset['annualized_return']:.2%}, Sharpe: {asset['sharpe_ratio']:.2f}, Signal: {signal_str}")
        
        # Print optimized weights
        print("\n=== Optimized Weights ===")
        for asset, weight in portfolio['optimized_weights'].items():
            print(f"{asset}: {weight:.2%}")
        
        # Print portfolio performance
        print("\n=== Portfolio Performance ===")
        print(f"Expected Return: {portfolio['performance']['return']:.2%}")
        print(f"Expected Volatility: {portfolio['performance']['volatility']:.2%}")
        print(f"Sharpe Ratio: {portfolio['performance']['sharpe_ratio']:.2f}")
        
        # Create visualizations
        allocation_chart, frontier_chart = system.visualize_portfolio(portfolio)
        
        # Show plots
        plt.show()
        
        # Save plots if output file specified
        if args.output:
            allocation_chart.savefig(f"{args.output}_allocation.png")
            frontier_chart.savefig(f"{args.output}_frontier.png")
            logger.info(f"Charts saved to {args.output}_allocation.png and {args.output}_frontier.png")
    
    except Exception as e:
        logger.error(f"Error creating portfolio: {e}")
        raise

def run_ml_command(system, args):
    """Run ML prediction command"""
    logger.info(f"Training ML model for {args.type}: {args.symbol}, prediction: {args.predict}, horizon: {args.horizon}")
    
    try:
        # Train model
        ml_results = system.train_ml_model(
            args.type,
            args.symbol,
            period=args.period,
            prediction_horizon=args.horizon,
            prediction_type=args.predict
        )
        
        # Print training results
        print("\n=== Model Training Results ===")
        if args.predict == 'direction':
            print(f"Accuracy: {ml_results['training_results']['accuracy']:.2%}")
            print(f"Precision: {ml_results['training_results']['precision']:.2%}")
            print(f"Recall: {ml_results['training_results']['recall']:.2%}")
            print(f"F1 Score: {ml_results['training_results']['f1_score']:.2f}")
        else:
            print(f"MSE: {ml_results['training_results']['mse']:.4f}")
            print(f"RMSE: {ml_results['training_results']['rmse']:.4f}")
            print(f"R² Score: {ml_results['training_results']['r2_score']:.2f}")
        
        # Print backtest results
        print("\n=== ML Strategy Backtest Results ===")
        print(f"Total Return: {ml_results['backtest_metrics']['total_return']:.2%}")
        print(f"Annualized Return: {ml_results['backtest_metrics']['annualized_return']:.2%}")
        print(f"Max Drawdown: {ml_results['backtest_metrics']['max_drawdown']:.2%}")
        print(f"Sharpe Ratio: {ml_results['backtest_metrics']['sharpe_ratio']:.2f}")
        
        # Print feature importance
        print("\n=== Top Feature Importance ===")
        top_features = ml_results['feature_importance'].head(10)
        for _, row in top_features.iterrows():
            print(f"{row['feature']}: {row['importance']:.4f}")
        
        # Create visualizations
        feature_chart = system.visualizer.plot_feature_importance(
            ml_results['feature_importance'],
            title=f"Feature Importance for {args.symbol} Prediction"
        )
        
        backtest_chart = system.visualizer.plot_backtest_results(
            ml_results['backtest_results'],
            ml_results['backtest_metrics'],
            title=f"ML Strategy Backtest for {args.symbol}"
        )
        
        # Show plots
        plt.show()
        
        # Save plots if output file specified
        if args.output:
            feature_chart.savefig(f"{args.output}_features.png")
            backtest_chart.savefig(f"{args.output}_ml_backtest.png")
            logger.info(f"Charts saved to {args.output}_features.png and {args.output}_ml_backtest.png")
    
    except Exception as e:
        logger.error(f"Error training ML model: {e}")
        raise

def run_risk_command(system, args):
    """Run risk analysis command"""
    logger.info(f"Analyzing portfolio risk for {args.symbols} over period {args.period}")
    
    try:
        # Analyze portfolio risk
        risk_results = system.analyze_portfolio_risk(
            args.symbols,
            period=args.period
        )
        
        # Print portfolio metrics
        portfolio_metrics = risk_results['portfolio_metrics'].iloc[0]
        
        print("\n=== Portfolio Risk Metrics ===")
        print(f"Annual Return: {portfolio_metrics['annual_return']:.2%}")
        print(f"Annual Volatility: {portfolio_metrics['annual_volatility']:.2%}")
        print(f"Sharpe Ratio: {portfolio_metrics['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio: {portfolio_metrics['sortino_ratio']:.2f}")
        print(f"Max Drawdown: {portfolio_metrics['max_drawdown']:.2%}")
        print(f"VaR (95%): {portfolio_metrics['var_95']:.2%}")
        print(f"CVaR (95%): {portfolio_metrics['cvar_95']:.2%}")
        
        # Create visualizations
        corr_chart = system.visualizer.plot_correlation_matrix(
            risk_results['correlation_matrix'],
            title="Asset Correlation Matrix"
        )
        
        mc_chart = system.visualizer.plot_monte_carlo_simulation(
            risk_results['simulation_results'],
            title="Portfolio Monte Carlo Simulation (1000 scenarios)"
        )
        
        # Show plots
        plt.show()
        
        # Save plots if output file specified
        if args.output:
            corr_chart.savefig(f"{args.output}_correlation.png")
            mc_chart.savefig(f"{args.output}_monte_carlo.png")
            logger.info(f"Charts saved to {args.output}_correlation.png and {args.output}_monte_carlo.png")
    
    except Exception as e:
        logger.error(f"Error analyzing portfolio risk: {e}")
        raise

if __name__ == "__main__":
    main()