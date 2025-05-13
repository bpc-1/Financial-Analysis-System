# utils.py - Utility functions

import pandas as pd
import numpy as np
import logging
import re
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("financial_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def normalize_dataframe(df, ticker=None):
    """Normalize dataframe column names to lowercase and handle special cases"""
    df = df.copy()
    
    # Handle potential MultiIndex columns (which come as tuples)
    if isinstance(df.columns, pd.MultiIndex):
        logger.info(f"Handling MultiIndex columns: {df.columns}")
        # For multi-index columns, take the first level which is the metric type
        new_cols = []
        for col in df.columns:
            if isinstance(col, tuple):
                # Take the first element (usually the metric like 'Close', 'Open', etc.)
                new_cols.append(str(col[0]).lower())
            else:
                new_cols.append(str(col).lower())
        df.columns = new_cols
    else:
        # For regular index, just lowercase
        df.columns = [str(col).lower() for col in df.columns]
    
    # Remove ticker suffix from columns if present
    if ticker:
        ticker_lower = ticker.lower()
        pattern = re.compile(f"(.+)_{ticker_lower}$")
        rename_dict = {}
        for col in df.columns:
            match = pattern.match(col)
            if match:
                rename_dict[col] = match.group(1)
        
        if rename_dict:
            logger.info(f"Renaming columns with ticker suffix: {rename_dict}")
            df = df.rename(columns=rename_dict)
    
    # Handle 'adj close' which becomes 'adj_close' or 'adj close'
    for adj_col in ['adj_close', 'adj close']:
        if adj_col in df.columns and 'close' not in df.columns:
            df = df.rename(columns={adj_col: 'close'})
    
    # Handle date column case sensitivity
    if 'Date' in df.columns and 'date' not in df.columns:
        df = df.rename(columns={'Date': 'date'})
    
    logger.info(f"Final normalized columns: {df.columns.tolist()}")
    return df

def calculate_sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
    """Calculate the Sharpe ratio of a returns series"""
    if len(returns) < 2:
        return 0
    
    excess_returns = returns - risk_free_rate / periods_per_year
    if excess_returns.std() == 0:
        return 0
    
    return np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()

def calculate_drawdown(equity_curve):
    """Calculate the drawdown series for an equity curve"""
    return 1 - equity_curve / equity_curve.cummax()