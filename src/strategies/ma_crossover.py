import pandas as pd
from typing import Dict, Any
from src.strategies.base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)

class MACrossoverStrategy(BaseStrategy):
    def __init__(self, short_window: int = 20, long_window: int = 50, warmup_period: int = 10):
        """
        Initialize strategy parameters
        
        Args:
            short_window: Period for the short moving average
            long_window: Period for the long moving average
            warmup_period: Number of data points to observe after both MAs are available before generating signals
                          This prevents taking trades based on the initial relationship between MAs
        """
        self.short_window = short_window
        self.long_window = long_window
        self.warmup_period = warmup_period
        super().__init__()
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate short and long moving averages"""
        # This method is kept for compatibility but not used directly
        return self.generate_signals(data)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on moving average crossover
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with signals (1 for buy, -1 for sell, 0 for hold)
        """
        # Make a copy to avoid modifying the original dataframe
        df = data.copy()
        
        # Handle both uppercase and lowercase column names
        close_col = 'Close' if 'Close' in df.columns else 'close'
        
        logger.debug(f"Using close column: {close_col}")
        logger.debug(f"Data shape: {df.shape}")
        
        # Calculate moving averages
        df['MA_Short'] = df[close_col].rolling(window=self.short_window).mean()
        df['MA_Long'] = df[close_col].rolling(window=self.long_window).mean()
        
        # Track previous signal state for detecting crossovers
        df['Prev_Signal'] = 0
        
        # Initialize signals column with zeros
        df['Signal'] = 0
        
        # Generate signals only where we have both moving averages
        valid_idx = df[['MA_Short', 'MA_Long']].dropna().index
        
        if len(valid_idx) > 0:
            logger.debug(f"Valid data points: {len(valid_idx)}")
            
            # Apply warmup period - skip initial data points
            if self.warmup_period > 0 and len(valid_idx) > self.warmup_period:
                # Identify first valid index where both MAs are available
                first_valid_idx = valid_idx[0]
                first_valid_pos = df.index.get_loc(first_valid_idx)
                
                # Apply warmup - find the index after the warmup period
                warmup_end_pos = first_valid_pos + self.warmup_period
                if warmup_end_pos < len(df):
                    warmup_end_idx = df.index[warmup_end_pos]
                    
                    # Set previous signals for the first valid point after warmup
                    if df.loc[warmup_end_idx, 'MA_Short'] > df.loc[warmup_end_idx, 'MA_Long']:
                        df.loc[warmup_end_idx, 'Prev_Signal'] = 1
                    else:
                        df.loc[warmup_end_idx, 'Prev_Signal'] = -1
                    
                    # Log warmup info
                    logger.info(f"Applied warmup period: skipping first {self.warmup_period} valid data points")
                    logger.info(f"First signal possible after: {warmup_end_idx}")
                    
                    # Only consider valid indices after warmup
                    valid_idx = valid_idx[valid_idx >= warmup_end_idx]
            
            # Handle case with no valid indices after warmup
            if len(valid_idx) == 0:
                logger.warning("No valid data points after warmup period")
                return df
            
            # Process each valid index after warmup
            prev_signal = df.loc[valid_idx[0], 'Prev_Signal']
            
            for i in range(len(valid_idx)):
                idx = valid_idx[i]
                
                # Set signal based on MA positions
                if df.loc[idx, 'MA_Short'] > df.loc[idx, 'MA_Long']:
                    current_signal = 1
                else:
                    current_signal = -1
                
                # Only generate signals on crossovers (signal change)
                if current_signal != prev_signal:
                    df.loc[idx, 'Signal'] = current_signal
                    prev_signal = current_signal
                
                # Store signal state for next iteration
                if i < len(valid_idx) - 1:
                    df.loc[valid_idx[i+1], 'Prev_Signal'] = current_signal
            
            # Log signal stats
            buy_signals = (df['Signal'] == 1).sum()
            sell_signals = (df['Signal'] == -1).sum()
            logger.debug(f"Generated {buy_signals} buy signals and {sell_signals} sell signals")
        else:
            logger.warning("No valid data for generating signals")
        
        return df 