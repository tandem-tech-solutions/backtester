#!/usr/bin/env python3
import os
import sys
import argparse
from src.config.config_manager import ConfigManager

def main():
    # Create argument parser first
    parser = argparse.ArgumentParser(description="Run a trading strategy backtest")
    
    # Add config file argument
    parser.add_argument('--config', type=str, default=None,
                        help="Path to configuration file (default: config.json in project root)")
    
    # Create the configuration manager
    config_manager = ConfigManager(config_path=parser.parse_known_args()[0].config)
    
    # Get available strategies
    available_strategies = config_manager.get_available_strategies()
    
    if not available_strategies:
        print("No strategies found in configuration file.")
        sys.exit(1)
    
    # Add more arguments
    parser.add_argument('--strategy', type=str, choices=list(available_strategies.keys()),
                        help="Strategy to run from config file")
    parser.add_argument('--days', type=int, default=180,
                        help="Number of days to backtest (default: 180)")
    parser.add_argument('--list', action='store_true',
                        help="List available strategies and exit")
    
    # Parse arguments
    args = parser.parse_args()
    
    # If --list is specified, print available strategies and exit
    if args.list:
        print("Available strategies:")
        for name, description in available_strategies.items():
            print(f"  {name}: {description}")
        sys.exit(0)
    
    # If no strategy specified, display available options and prompt user
    if not args.strategy:
        print("Please select a strategy:")
        for i, (name, description) in enumerate(available_strategies.items(), 1):
            print(f"  {i}. {name}: {description}")
        
        # Get user selection
        try:
            selection = int(input("\nEnter selection (number): "))
            if selection < 1 or selection > len(available_strategies):
                print("Invalid selection.")
                sys.exit(1)
                
            # Convert selection to strategy name
            strategy_name = list(available_strategies.keys())[selection - 1]
        except (ValueError, IndexError):
            print("Invalid selection.")
            sys.exit(1)
    else:
        strategy_name = args.strategy
        
    # Run the backtest
    print(f"Running backtest with strategy: {strategy_name}")
    print(f"Backtest period: {args.days} days")
    
    # Get strategy parameters from config
    strategy_params = config_manager.get_strategy_params(strategy_name)
    
    # Get base strategy type - this is what determines the implementation to use
    base_strategy_type = strategy_params.get('strategy_type', strategy_name)
    
    # Set environment variables for the backtest
    os.environ['STRATEGY_TYPE'] = base_strategy_type
    os.environ['BACKTEST_DAYS'] = str(args.days)
    
    # IMPORTANT: Also set the STRATEGY_NAME environment variable to record which config we're using
    os.environ['STRATEGY_NAME'] = strategy_name
    
    # Set environment variables for all strategy parameters
    for param, value in strategy_params.items():
        if value is not None:  # Skip null values
            os.environ[param] = str(value)
            print(f"Setting parameter {param}={value}")
    
    # Import and run the example module
    from src.example import main as run_backtest
    run_backtest()

if __name__ == "__main__":
    main() 