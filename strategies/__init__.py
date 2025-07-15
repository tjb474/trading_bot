from .ml_moving_average_crossover import MLMovingAverageCrossover
from .ml_open_range_breakout import MLOpenRangeBreakout

def get_strategy(strategy_name: str):
    """
    Factory function to return the appropriate strategy class.
    
    Args:
        strategy_name (str): Name of the strategy to return
        
    Returns:
        Strategy class that can be used with backtrader
    """
    strategies = {
        'ml_moving_average_crossover': MLMovingAverageCrossover,
        'ml_open_range_breakout': MLOpenRangeBreakout
    }
    
    if strategy_name not in strategies:
        raise ValueError(f"Strategy '{strategy_name}' not found. Available strategies: {list(strategies.keys())}")
        
    return strategies[strategy_name]

