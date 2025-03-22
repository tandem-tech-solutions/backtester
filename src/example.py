import logging
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import csv
import numpy as np
from adjustText import adjust_text

from src.data.dhan_data import DhanDataFetcher
from src.strategies.ma_crossover import MACrossoverStrategy
from src.backtester.backtester import Backtester

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration from environment variables with defaults
SHORT_MA_PERIOD = int(os.environ.get('SHORT_MA_PERIOD', 20))
LONG_MA_PERIOD = int(os.environ.get('LONG_MA_PERIOD', 50))
WARMUP_PERIOD = int(os.environ.get('WARMUP_PERIOD', 15))
BACKTEST_DAYS = int(os.environ.get('BACKTEST_DAYS', 365))

def save_trade_journal(trades, strategy_params, output_file="trade_journal.csv"):
    """Save trade data to a CSV file with strategy parameters"""
    if not trades:
        logger.warning("No trades to save")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    # Prepare data for CSV
    trade_data = []
    for trade in trades:
        # Add strategy parameters to each trade
        trade_with_params = trade.copy()
        trade_with_params.update(strategy_params)
        trade_data.append(trade_with_params)
    
    # Write to CSV
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = list(trade_data[0].keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trade_data)
    
    logger.info(f"Trade journal saved to {output_file}")

def plot_portfolio_with_annotations(portfolio_df, trades, strategy_params, max_drawdown, data_with_indicators=None, output_file="backtest_results.png"):
    """Create an enhanced plot with trade annotations, strategy parameters, and indicators"""
    # Create figure with subplots - price/indicators on top, portfolio value in middle, trade table at bottom
    fig = plt.figure(figsize=(14, 16))
    
    # Define grid for the three main components
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1.2], hspace=0.3)
    
    # Price and indicators subplot
    ax1 = fig.add_subplot(gs[0])
    
    # Portfolio value subplot
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    
    # Trade table subplot
    ax3 = fig.add_subplot(gs[2])
    ax3.axis('off')  # Hide axes for table
    
    # Plot price and indicators on top subplot
    if data_with_indicators is not None:
        # Get column names - handle both uppercase and lowercase
        close_col = 'Close' if 'Close' in data_with_indicators.columns else 'close'
        
        # Plot price
        ax1.plot(data_with_indicators.index, data_with_indicators[close_col], label='Price', color='black', alpha=0.7)
        
        # Plot moving averages
        ax1.plot(data_with_indicators.index, data_with_indicators['MA_Short'], label=f'MA Short ({strategy_params["short_window"]})', color='green', linewidth=1.5)
        ax1.plot(data_with_indicators.index, data_with_indicators['MA_Long'], label=f'MA Long ({strategy_params["long_window"]})', color='red', linewidth=1.5)
        
        # Add buy/sell markers at the price level
        for trade in trades:
            if trade['position'] == 1:  # Buy signal
                ax1.scatter(trade['entry_date'], trade['entry_price'], marker='^', color='green', s=100)
            else:  # Sell signal
                ax1.scatter(trade['entry_date'], trade['entry_price'], marker='v', color='red', s=100)
        
        # Customize price subplot
        ax1.set_title(f'Price and Moving Averages - {strategy_params["symbol"]}', fontsize=12)
        ax1.set_ylabel('Price (₹)', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        
        # Set y-axis to show a bit of padding above/below the price range
        y_min = data_with_indicators[close_col].min() * 0.98
        y_max = data_with_indicators[close_col].max() * 1.02
        ax1.set_ylim(y_min, y_max)
    
    # Plot portfolio value on middle subplot
    ax2.plot(portfolio_df.index, portfolio_df['Portfolio_Value'], label='Portfolio Value', linewidth=2, color='blue')
    
    # Add horizontal line for initial capital
    ax2.axhline(y=portfolio_df['Portfolio_Value'].iloc[0], color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
    
    # Prepare for annotations - just simple markers, no text annotations on the chart
    buy_dates, buy_values = [], []
    sell_dates, sell_values = [], []
    
    # Calculate total PNL to match final result
    total_pnl = portfolio_df['Portfolio_Value'].iloc[-1] - portfolio_df['Portfolio_Value'].iloc[0]
    total_pnl_percentage = ((portfolio_df['Portfolio_Value'].iloc[-1] / portfolio_df['Portfolio_Value'].iloc[0]) - 1) * 100
    
    for trade in trades:
        # Entry points
        entry_date = trade['entry_date']
        entry_value = portfolio_df.loc[entry_date, 'Portfolio_Value']
        
        # Add to collections for scatter plot
        if trade['position'] == 1:
            buy_dates.append(entry_date)
            buy_values.append(entry_value)
        else:
            sell_dates.append(entry_date)
            sell_values.append(entry_value)
        
        # Exit points
        exit_date = trade['exit_date']
        exit_value = portfolio_df.loc[exit_date, 'Portfolio_Value']
        
        # Add to collections for scatter plot
        if trade['position'] == 1:
            sell_dates.append(exit_date)
            sell_values.append(exit_value)
        else:
            buy_dates.append(exit_date)
            buy_values.append(exit_value)
    
    # Add scatter plots for buy/sell points with numbers for reference
    if buy_dates:
        for i, (date, value) in enumerate(zip(buy_dates, buy_values)):
            ax2.scatter(date, value, color='green', s=80, marker='^')
            # Small number label next to marker
            ax2.text(date, value*1.01, str(i+1), fontsize=8, ha='center')
    
    if sell_dates:
        for i, (date, value) in enumerate(zip(sell_dates, sell_values)):
            ax2.scatter(date, value, color='red', s=80, marker='v')
            # Small number label next to marker
            ax2.text(date, value*0.99, str(i+1), fontsize=8, ha='center')
    
    # Customize portfolio subplot
    ax2.set_title('Portfolio Value Over Time', fontsize=12)
    ax2.set_ylabel('Portfolio Value (₹)', fontsize=10)
    ax2.set_xlabel('Date', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    
    # Format x-axis dates
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # Create detailed trade table at the bottom
    # Prepare table data
    table_data = []
    headers = ['#', 'Type', 'Entry Date', 'Entry Price', 'Exit Date', 'Exit Price', 'PnL (₹)', 'Return (%)', 'Cumulative PnL (₹)']
    cumulative_pnl = 0
    
    for i, trade in enumerate(trades):
        trade_type = "Long" if trade['position'] == 1 else "Short"
        entry_date = trade['entry_date'].strftime('%Y-%m-%d')
        exit_date = trade['exit_date'].strftime('%Y-%m-%d')
        pnl = trade['pnl']
        cumulative_pnl += pnl
        trade_return = trade['return'] * 100  # Convert to percentage
        
        row = [
            i+1,
            trade_type,
            entry_date,
            f"₹{trade['entry_price']:.2f}",
            exit_date,
            f"₹{trade['exit_price']:.2f}",
            f"₹{pnl:.2f}",
            f"{trade_return:.2f}%",
            f"₹{cumulative_pnl:.2f}"
        ]
        table_data.append(row)
    
    # Create and style the table
    table = ax3.table(
        cellText=table_data,
        colLabels=headers,
        loc='center',
        cellLoc='center',
        colColours=['#f2f2f2']*len(headers),
        bbox=[0.05, 0.2, 0.9, 0.7]  # [left, bottom, width, height]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)  # Adjust row height
    
    # Style PnL cells based on positive/negative values
    for i, row in enumerate(table_data):
        pnl_value = float(row[6].replace('₹', ''))
        if pnl_value > 0:
            table[(i+1, 6)].set_facecolor('#d4f7d4')  # Light green for profit
        elif pnl_value < 0:
            table[(i+1, 6)].set_facecolor('#f7d4d4')  # Light red for loss
    
    # Add table title
    ax3.set_title('Trade Details', fontsize=12)
    
    # Add strategy info and stats to bottom right
    stats_text = (
        f"Strategy Parameters:\n"
        f"- Symbol: {strategy_params['symbol']}\n"
        f"- Short MA: {strategy_params['short_window']} days\n"
        f"- Long MA: {strategy_params['long_window']} days\n"
        f"- Warmup Period: {strategy_params['warmup_period']} days\n"
        f"- Initial Capital: ₹{strategy_params['initial_capital']:.2f}\n"
        f"- Commission: {strategy_params['commission']:.3f}\n\n"
        f"Backtest Results:\n"
        f"- Final Value: ₹{portfolio_df['Portfolio_Value'].iloc[-1]:.2f}\n"
        f"- Total Return: {total_pnl_percentage:.2f}% (₹{total_pnl:.2f})\n"
        f"- Max Drawdown: {max_drawdown:.2f}%\n"
        f"- Number of Trades: {len(trades)}\n"
        f"- Date Range: {strategy_params['start_date']} to {strategy_params['end_date']}"
    )
    
    # Add text box with stats in the bottom right corner
    plt.figtext(0.55, 0.02, stats_text, fontsize=9, 
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=1'))
    
    # Add overall title
    fig.suptitle(f'Backtest Results: MA Crossover Strategy ({strategy_params["short_window"]}/{strategy_params["long_window"]})', 
                fontsize=16, y=0.98)
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Enhanced plot with indicators saved to {output_file}")
    plt.close()

def main():
    # Initialize data fetcher
    data_fetcher = DhanDataFetcher()

    # Set date range for historical data using configurable duration
    end_date = datetime.now()  # End date is current date
    start_date = end_date - timedelta(days=BACKTEST_DAYS)
    
    logger.info(f"Fetching data for date range: {start_date.date()} to {end_date.date()}")
    logger.info(f"Using {BACKTEST_DAYS} days of historical data")

    # Fetch historical data for RELIANCE
    symbol = "RELIANCE"
    logger.info(f"Fetching data for symbol: {symbol}")
    
    historical_data = data_fetcher.get_historical_data(symbol, start_date, end_date)
    
    if historical_data is None:
        logger.error("Failed to fetch historical data")
        return
    
    if len(historical_data) == 0:
        logger.error("No historical data found for the specified date range")
        return
        
    logger.info(f"Successfully fetched {len(historical_data)} data points")
    logger.info(f"Data sample:\n{historical_data.head()}")

    # Get current price
    current_price = data_fetcher.get_current_price(symbol)
    logger.info(f"Current price for {symbol}: {current_price}")

    # Initialize strategy with environment variable parameters
    short_window = SHORT_MA_PERIOD
    long_window = LONG_MA_PERIOD
    warmup_period = WARMUP_PERIOD
    logger.info(f"Using strategy parameters: short_window={short_window}, long_window={long_window}, warmup_period={warmup_period}")
    
    strategy = MACrossoverStrategy(short_window=short_window, long_window=long_window, warmup_period=warmup_period)
    
    # Create strategy parameters dictionary
    strategy_params = {
        'symbol': symbol,
        'short_window': short_window,
        'long_window': long_window,
        'warmup_period': warmup_period,
        'start_date': start_date.strftime('%Y-%m-%d'),
        'end_date': end_date.strftime('%Y-%m-%d'),
        'initial_capital': 100000,
        'commission': 0.001
    }

    # Initialize and run backtester
    backtester = Backtester(initial_capital=strategy_params['initial_capital'], 
                            commission=strategy_params['commission'])
    results = backtester.run(strategy, historical_data)

    # Print results
    print("\nBacktest Results:")
    print(f"Final Portfolio Value: ₹{results['final_portfolio_value']:.2f}")
    print(f"Total Return: {results['total_return']:.2f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
    print(f"Number of Trades: {results['n_trades']}")
    
    # Save trade journal
    save_trade_journal(results['trades'], strategy_params, 'trade_journal.csv')

    # Get data with indicators for plotting
    data_with_indicators = strategy.generate_signals(historical_data.copy())
    
    # Create enhanced portfolio plot with annotations and indicators
    plot_portfolio_with_annotations(
        results['portfolio_df'], 
        results['trades'], 
        strategy_params, 
        results['max_drawdown'],
        data_with_indicators,
        'backtest_results.png'
    )

if __name__ == "__main__":
    main()