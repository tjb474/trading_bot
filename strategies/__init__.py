from .ml_moving_average_crossover import MLMovingAverageCrossover
# from .rsi_mean_reversion import RsiMeanReversion # Uncomment when you implement this

def get_strategy(strategy_name: str):
    """
    Factory function to get a strategy class by its name.
    """
    strategies = {
        "ml_moving_average_crossover": MLMovingAverageCrossover,
        # "rsi_mean_reversion": RsiMeanReversion,
    }
    strategy_class = strategies.get(strategy_name.lower())
    if not strategy_class:
        raise ValueError(f"Strategy '{strategy_name}' not found.")
    return strategy_class