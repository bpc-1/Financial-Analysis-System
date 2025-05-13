# README.md - Financial Analysis System

# Advanced Financial Analysis System

An expert-level financial analysis platform capable of sophisticated technical analysis, portfolio optimization, and machine learning-based predictions for stocks and cryptocurrencies.

## Features

- **Comprehensive Technical Analysis**: Over 15 technical indicators including Moving Averages, MACD, RSI, Bollinger Bands, Ichimoku Cloud, and more
- **Advanced Signal Generation**: Multi-factor signal generation with customizable weights
- **Portfolio Optimization**: Modern Portfolio Theory implementation with efficient frontier generation
- **Machine Learning Predictions**: Price direction and value forecasting with feature importance analysis
- **Risk Analysis**: Monte Carlo simulations, correlation analysis, and comprehensive risk metrics
- **Interactive Dashboard**: Streamlit-based dashboard for easy visualization and analysis
- **Alternative Data Integration**: Sentiment analysis and volume indicators
- **Command-line Interface**: Scriptable analysis for automation

## Installation

### Requirements

- Python 3.8+
- Required packages: pandas, numpy, matplotlib, scikit-learn, scipy, streamlit, yfinance, and more

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/financial-analysis-system.git
cd financial-analysis-system
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

The system provides a command-line interface for various analyses:

#### Analyze a Single Asset

```bash
python analyze.py analyze --type stock --symbol AAPL --period 1y --sentiment
```

Options:
- `--type`, `-t`: Asset type (`stock` or `crypto`)
- `--symbol`, `-s`: Asset symbol/ticker
- `--period`, `-p`: Analysis period (`1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y`)
- `--sentiment`: Include sentiment analysis
- `--output`, `-o`: Output file for charts

#### Create Optimized Portfolio

```bash
python analyze.py portfolio --type stock --count 5 --period 3mo
```

Options:
- `--type`, `-t`: Asset type (`stock` or `crypto`)
- `--count`, `-c`: Number of assets in portfolio
- `--period`, `-p`: Analysis period
- `--output`, `-o`: Output file for charts

#### Train ML Model for Prediction

```bash
python analyze.py ml --type stock --symbol AAPL --predict direction --horizon 5
```

Options:
- `--type`, `-t`: Asset type (`stock` or `crypto`)
- `--symbol`, `-s`: Asset symbol/ticker
- `--predict`, `-p`: Prediction type (`direction`, `close`, `returns`)
- `--horizon`: Prediction horizon in days
- `--period`: Training period
- `--output`, `-o`: Output file for charts

#### Analyze Portfolio Risk

```bash
python analyze.py risk --symbols AAPL MSFT GOOGL --period 1y
```

Options:
- `--symbols`, `-s`: Space-separated list of asset symbols
- `--period`, `-p`: Analysis period
- `--output`, `-o`: Output file for charts

### Interactive Dashboard

To launch the interactive dashboard:

```bash
streamlit run dashboard.py
```

The dashboard provides a user-friendly interface for:
- Asset Analysis
- Portfolio Optimization
- ML Predictions
- Risk Analysis

## Project Structure

```
financial-analysis-system/
├── analyze.py           # Command-line interface
├── dashboard.py         # Streamlit dashboard
├── config.py            # Configuration settings
├── utils.py             # Utility functions
├── data_fetcher.py      # Data gathering component
├── technical_analysis.py # Technical indicators
├── signal_generator.py  # Trading signal generation
├── portfolio_optimizer.py # Portfolio optimization
├── backtester.py        # Strategy backtesting
├── ml_predictor.py      # Machine learning component
├── visualizer.py        # Visualization utilities
├── financial_system.py  # Main system integration
└── requirements.txt     # Dependencies
```

## Customization

### Adding New Technical Indicators

Extend the `TechnicalAnalyzer` class in `technical_analysis.py` with new indicators:

```python
def calculate_your_indicator(self, df, param1=default1, param2=default2):
    """Calculate your custom indicator"""
    result = df.copy()
    # Implementation...
    result['your_indicator'] = calculated_values
    return result
```

### Creating Custom Trading Strategies

Add methods to the `SignalGenerator` class in `signal_generator.py`:

```python
def generate_your_signal(self, df, param1=default1):
    """Generate signals based on your criteria"""
    result = df.copy()
    result['your_signal'] = 0
    # Implementation...
    return result
```

## Further Development

Areas for enhancement:
- Integration with real-time data providers
- Support for options and futures
- Addition of fundamental analysis metrics
- Neural network-based deep learning models
- Trade execution via broker APIs
- Multiple timeframe analysis

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational and informational purposes only. It is not intended as financial advice or a recommendation to trade. Trading financial instruments carries significant risk, including the potential for loss of funds. Past performance is not indicative of future results.