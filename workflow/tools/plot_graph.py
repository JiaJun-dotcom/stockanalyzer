import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any

def plot_stock_data(data: pd.DataFrame, ticker: str, indicators: Optional[List[str]] = None, 
                   save_path: Optional[str] = None, show_plot: bool = True):
    """
    Plot stock data with technical indicators
    
    Args:
        data: DataFrame containing stock data with technical indicators
        ticker: Stock ticker symbol
        indicators: List of indicators to plot (default: ['EMA_200', 'RSI', 'MACD', 'BB_High', 'BB_Low'])
        save_path: Path to save the plot (optional)
        show_plot: Whether to display the plot (default: True)
        
    Returns:
        Dictionary with figure and axes objects
    """
    if indicators is None:
        indicators = ['EMA_200', 'RSI', 'MACD', 'BB_High', 'BB_Low']
    
    # Validate that required columns exist
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Required column '{col}' not found in data")
    
    # Create figure and subplots
    fig = plt.figure(figsize=(14, 10))
    
    # Define grid layout based on indicators
    grid_spec = plt.GridSpec(4, 1, height_ratios=[3, 1, 1, 1])
    
    # Main price chart
    ax1 = fig.add_subplot(grid_spec[0])
    ax1.set_title(f"{ticker} Stock Price with Technical Indicators")
    
    # Plot candlestick chart
    width = 0.6
    width2 = 0.05
    up = data[data.Close >= data.Open]
    down = data[data.Close < data.Open]
    
    # Plot up candles
    ax1.bar(up.index, up.Close-up.Open, width, bottom=up.Open, color='green')
    ax1.bar(up.index, up.High-up.Close, width2, bottom=up.Close, color='green')
    ax1.bar(up.index, up.Low-up.Open, width2, bottom=up.Open, color='green')
    
    # Plot down candles
    ax1.bar(down.index, down.Close-down.Open, width, bottom=down.Open, color='red')
    ax1.bar(down.index, down.High-down.Open, width2, bottom=down.Open, color='red')
    ax1.bar(down.index, down.Low-down.Close, width2, bottom=down.Close, color='red')
    
    # Add technical indicators to price chart
    if 'EMA_200' in indicators and 'EMA_200' in data.columns:
        ax1.plot(data.index, data['EMA_200'], color='blue', linewidth=1.5, label='EMA 200')
    
    if 'BB_High' in indicators and 'BB_High' in data.columns and 'BB_Low' in data.columns:
        ax1.plot(data.index, data['BB_High'], 'k--', linewidth=1, label='Bollinger High')
        ax1.plot(data.index, data['BB_Low'], 'k--', linewidth=1, label='Bollinger Low')
    
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Volume subplot
    ax2 = fig.add_subplot(grid_spec[1], sharex=ax1)
    ax2.bar(data.index, data['Volume'], color='purple', alpha=0.5)
    ax2.set_ylabel('Volume')
    ax2.grid(True, alpha=0.3)
    
    # RSI subplot
    if 'RSI' in indicators and 'RSI' in data.columns:
        ax3 = fig.add_subplot(grid_spec[2], sharex=ax1)
        ax3.plot(data.index, data['RSI'], color='orange', linewidth=1.5)
        ax3.axhline(70, color='red', linestyle='--', alpha=0.5)
        ax3.axhline(30, color='green', linestyle='--', alpha=0.5)
        ax3.set_ylabel('RSI')
        ax3.set_ylim(0, 100)
        ax3.grid(True, alpha=0.3)
    
    # MACD subplot
    if 'MACD' in indicators and 'MACD' in data.columns and 'MACD_Signal' in data.columns:
        ax4 = fig.add_subplot(grid_spec[3], sharex=ax1)
        ax4.plot(data.index, data['MACD'], color='blue', linewidth=1.5, label='MACD')
        ax4.plot(data.index, data['MACD_Signal'], color='red', linewidth=1.5, label='Signal')
        
        # Plot MACD histogram if available
        if 'MACD_Hist' in data.columns:
            ax4.bar(data.index, data['MACD_Hist'], color='green', alpha=0.5, label='Histogram')
        
        ax4.legend(loc='upper left')
        ax4.set_ylabel('MACD')
        ax4.grid(True, alpha=0.3)
    
    # Format x-axis dates
    for ax in fig.get_axes():
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show plot if requested
    if show_plot:
        plt.show()
    
    return {
        'figure': fig,
        'axes': fig.get_axes()
    }

def plot_comparison(data_dict: Dict[str, pd.DataFrame], metric: str = 'Close', 
                    title: str = 'Stock Comparison', save_path: Optional[str] = None, 
                    show_plot: bool = True):
    """
    Plot comparison of multiple stocks for a given metric
    
    Args:
        data_dict: Dictionary mapping ticker symbols to DataFrames
        metric: Column name to compare (default: 'Close')
        title: Plot title
        save_path: Path to save the plot (optional)
        show_plot: Whether to display the plot (default: True)
        
    Returns:
        Dictionary with figure and axes objects
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for ticker, data in data_dict.items():
        if metric not in data.columns:
            print(f"Warning: Metric '{metric}' not found for {ticker}. Skipping.")
            continue
        
        # Normalize to percentage change from first day for fair comparison
        first_value = data[metric].iloc[0]
        normalized = (data[metric] / first_value - 1) * 100
        ax.plot(data.index, normalized, linewidth=2, label=ticker)
    
    ax.set_title(title)
    ax.set_ylabel(f'Percentage Change in {metric}')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show plot if requested
    if show_plot:
        plt.show()
    
    return {
        'figure': fig,
        'axes': ax
    }

if __name__ == "__main__":
    # Example usage
    from workflow.tools.get_data_with_indicators import get_data_with_indicators
    
    # Get data for a single stock
    ticker = "AAPL"
    data = get_data_with_indicators(ticker=ticker, start_date="2022-01-01", end_date="2023-01-01")
    
    # Plot stock data with indicators
    plot_stock_data(data, ticker)
    
    # Example of comparing multiple stocks
    tickers = ["AAPL", "MSFT", "GOOG"]
    data_dict = {}
    
    for ticker in tickers:
        data_dict[ticker] = get_data_with_indicators(
            ticker=ticker, 
            start_date="2022-01-01", 
            end_date="2023-01-01"
        )
    
    # Plot comparison
    plot_comparison(data_dict, title="Tech Stock Comparison")