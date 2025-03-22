import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class Backtester:
    """Backtester for trading strategies"""
    
    def __init__(self, initial_capital: float = 100000, commission: float = 0.001):
        """Initialize backtester"""
        self.initial_capital = initial_capital
        self.commission = commission
    
    def run(self, strategy, data):
        """
        Run backtest
        
        Args:
            strategy: Strategy instance
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with backtest results
        """
        # Rename columns to match expected format
        data_copy = data.copy()
        if 'close' in data_copy.columns:
            data_copy.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }, inplace=True)
        
        logger.debug(f"Running backtest on data with shape: {data_copy.shape}")
        
        # Generate signals using the strategy
        data_with_signals = strategy.generate_signals(data_copy)
        
        # Log signal statistics
        buy_signals = (data_with_signals['Signal'] == 1).sum()
        sell_signals = (data_with_signals['Signal'] == -1).sum()
        logger.debug(f"Signal counts: Buy={buy_signals}, Sell={sell_signals}, Hold={(data_with_signals['Signal'] == 0).sum()}")
        
        # Initialize portfolio metrics
        cash = self.initial_capital
        position = 0  # 0 for no position, 1 for long, -1 for short
        position_size = 0
        position_value = 0
        trades = []
        
        # Create dataframe to track portfolio
        portfolio_df = pd.DataFrame(index=data_with_signals.index)
        portfolio_df['Signal'] = data_with_signals['Signal']
        portfolio_df['Price'] = data_with_signals['Close']
        portfolio_df['Cash'] = 0.0
        portfolio_df['Position'] = 0
        portfolio_df['Position_Size'] = 0.0
        portfolio_df['Position_Value'] = 0.0
        portfolio_df['Portfolio_Value'] = 0.0
        
        # Run backtest - iterate through each day
        for i in range(1, len(data_with_signals)):
            current_date = data_with_signals.index[i]
            prev_date = data_with_signals.index[i-1]
            
            # Get current and previous data
            current = data_with_signals.iloc[i]
            prev = data_with_signals.iloc[i-1]
            
            # Check for signal changes
            if current['Signal'] != prev['Signal']:
                # Close existing position if any
                if position != 0:
                    # Calculate trade PnL
                    exit_price = current['Open']
                    
                    if position == 1:  # Long position
                        # For longs: PnL = (exit_price - entry_price) * position_size - commission
                        trade_value = position_size * exit_price
                        commission_cost = trade_value * self.commission
                        trade_pnl = (exit_price - position_entry_price) * position_size - commission_cost
                        logger.info(f"LONG EXIT - Entry: {position_entry_price}, Exit: {exit_price}, Size: {position_size}, "
                                   f"Calculation: ({exit_price} - {position_entry_price}) * {position_size} - commission = {trade_pnl:.2f}")
                    else:  # Short position
                        # For shorts: PnL = (entry_price - exit_price) * position_size - commission
                        trade_value = position_size * exit_price
                        commission_cost = trade_value * self.commission
                        trade_pnl = (position_entry_price - exit_price) * position_size - commission_cost
                    
                    # Update cash
                    if position == 1:
                        cash += trade_value - commission_cost  # Add exit value minus commission
                    else:
                        cash += (position_entry_price * position_size) - trade_value - commission_cost  # Return collateral + profit (or - loss)
                    
                    # Calculate portfolio contribution percentage
                    portfolio_value_before_exit = cash - trade_pnl
                    portfolio_return = trade_pnl / portfolio_value_before_exit
                    
                    # Record trade
                    trades.append({
                        'entry_date': position_entry_date,
                        'exit_date': current_date,
                        'entry_price': position_entry_price,
                        'exit_price': exit_price,
                        'position': position,
                        'position_size': position_size,
                        'pnl': trade_pnl,
                        'return': portfolio_return  # This is now the percentage impact on total portfolio
                    })
                    
                    logger.debug(f"Closed position at {current_date}: Type={'Long' if position == 1 else 'Short'}, "
                                 f"Size={position_size}, PnL={trade_pnl:.2f}, Portfolio Impact={portfolio_return*100:.2f}%")
                    
                    # Reset position
                    position = 0
                    position_size = 0
                    position_value = 0
                
                # Open new position based on signal
                if current['Signal'] != 0:
                    entry_price = current['Open']
                    
                    # Use 100% of available cash for each trade instead of 20%
                    available_cash = cash
                    new_position_size = int(available_cash / entry_price)
                    
                    # Ensure we have at least 1 share
                    if new_position_size < 1:
                        new_position_size = 1 if cash >= entry_price else 0
                    
                    if new_position_size > 0:
                        position = 1 if current['Signal'] == 1 else -1
                        position_size = new_position_size
                        trade_value = position_size * entry_price
                        trade_commission = trade_value * self.commission
                        
                        # For long positions, we spend cash
                        if position == 1:
                            cash -= (trade_value + trade_commission)
                            position_value = trade_value  # For long positions, this is current value
                        # For short positions, we get cash but pay commission
                        else:
                            cash -= trade_commission  # Only pay commission
                            position_value = trade_value  # Store the entry value for proper PnL calculation
                        
                        position_entry_date = current_date
                        position_entry_price = entry_price
                        
                        logger.debug(f"Opened position at {current_date}: Type={'Long' if position == 1 else 'Short'}, "
                                    f"Size={position_size}, Entry Price={entry_price:.2f}")
            
            # Calculate current position value
            if position != 0:
                current_price = current['Close']
                
                if position == 1:  # Long position
                    # For long positions, value changes with current price
                    position_value = position_size * current_price
                # For short positions, we maintain position_value at entry value for correct PnL calculation
            
            # Calculate total portfolio value
            if position == 1:  # Long position
                portfolio_value = cash + position_value
            elif position == -1:  # Short position
                # For shorts, portfolio value = cash + unrealized PnL
                # Unrealized PnL = (entry_price - current_price) * position_size
                unrealized_pnl = (position_entry_price - current['Close']) * position_size
                portfolio_value = cash + unrealized_pnl
            else:  # No position
                portfolio_value = cash
            
            # Update portfolio dataframe
            portfolio_df.loc[current_date, 'Cash'] = cash
            portfolio_df.loc[current_date, 'Position'] = position
            portfolio_df.loc[current_date, 'Position_Size'] = position_size
            portfolio_df.loc[current_date, 'Position_Value'] = position_value
            portfolio_df.loc[current_date, 'Portfolio_Value'] = portfolio_value
            
            # Add debug logging about position value calculation
            if position != 0:
                logger.info(f"Date: {current_date}, Position: {'LONG' if position == 1 else 'SHORT'}, Size: {position_size}, "
                          f"Entry Price: {position_entry_price}, Current Price: {current['Close']}, "
                          f"Position Value: {position_value:.2f}")
        
        # Close any open position at the end
        if position != 0:
            exit_price = data_with_signals.iloc[-1]['Close']
            
            if position == 1:  # Long position
                # For longs: PnL = (exit_price - entry_price) * position_size - commission
                trade_value = position_size * exit_price
                commission_cost = trade_value * self.commission
                trade_pnl = (exit_price - position_entry_price) * position_size - commission_cost
            else:  # Short position
                # For shorts: PnL = (entry_price - exit_price) * position_size - commission
                trade_value = position_size * exit_price
                commission_cost = trade_value * self.commission
                trade_pnl = (position_entry_price - exit_price) * position_size - commission_cost
            
            # Update cash
            if position == 1:
                cash += trade_value - commission_cost  # Add exit value minus commission
            else:
                cash += (position_entry_price * position_size) - trade_value - commission_cost  # Return collateral + profit (or - loss)
            
            # Calculate portfolio contribution percentage
            portfolio_value_before_exit = cash - trade_pnl
            portfolio_return = trade_pnl / portfolio_value_before_exit
            
            # Record trade
            trades.append({
                'entry_date': position_entry_date,
                'exit_date': data_with_signals.index[-1],
                'entry_price': position_entry_price,
                'exit_price': exit_price,
                'position': position,
                'position_size': position_size,
                'pnl': trade_pnl,
                'return': portfolio_return  # This is now the percentage impact on total portfolio
            })
            
            logger.debug(f"Closed final position: Type={'Long' if position == 1 else 'Short'}, "
                         f"Size={position_size}, PnL={trade_pnl:.2f}, Portfolio Impact={portfolio_return*100:.2f}%")
            
            # Update final portfolio value
            portfolio_value = cash
            portfolio_df.iloc[-1, portfolio_df.columns.get_loc('Portfolio_Value')] = portfolio_value
        
        # Fill first row of portfolio dataframe
        portfolio_df.iloc[0, portfolio_df.columns.get_loc('Cash')] = self.initial_capital
        portfolio_df.iloc[0, portfolio_df.columns.get_loc('Portfolio_Value')] = self.initial_capital
        
        # Forward fill any missing values
        portfolio_df = portfolio_df.fillna(method='ffill')
        
        # Calculate metrics
        initial_value = portfolio_df['Portfolio_Value'].iloc[0]
        final_value = portfolio_df['Portfolio_Value'].iloc[-1]
        
        # Calculate total return as percentage
        total_return = ((final_value / initial_value) - 1) * 100
        
        # Calculate daily returns
        portfolio_df['Daily_Return'] = portfolio_df['Portfolio_Value'].pct_change()
        
        # Calculate annualized Sharpe ratio (assuming 252 trading days per year and risk-free rate of 0)
        daily_returns = portfolio_df['Daily_Return'].dropna()
        
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
        else:
            sharpe_ratio = 0
        
        # Calculate maximum drawdown
        portfolio_df['Cummax'] = portfolio_df['Portfolio_Value'].cummax()
        portfolio_df['Drawdown'] = (portfolio_df['Portfolio_Value'] - portfolio_df['Cummax']) / portfolio_df['Cummax'] * 100
        max_drawdown = portfolio_df['Drawdown'].min()
        
        logger.info(f"Backtest completed: Return={total_return:.2f}%, Sharpe={sharpe_ratio:.2f}, Drawdown={max_drawdown:.2f}%")
        
        return {
            'final_portfolio_value': final_value,
            'total_return': total_return,  # Already in percentage
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,  # Already in percentage
            'n_trades': len(trades),
            'trades': trades,
            'portfolio_values': portfolio_df['Portfolio_Value'],
            'portfolio_df': portfolio_df  # Return full portfolio dataframe for analysis
        }