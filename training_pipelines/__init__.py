from .ma_crossover_pipeline import MovingAverageCrossoverPipeline
from .open_range_breakout_pipeline import OpenRangeBreakoutPipeline

def get_training_pipeline(pipeline_name: str):
    """
    Factory function to get a training pipeline class by its name.
    """
    pipelines = {
        "ml_moving_average_crossover": MovingAverageCrossoverPipeline,
        "ml_open_range_breakout": OpenRangeBreakoutPipeline,
    }
    pipeline_class = pipelines.get(pipeline_name.lower())
    
    if not pipeline_class:
        raise ValueError(f"Training pipeline '{pipeline_name}' not found.")
        
    # Return an INSTANCE of the class
    return pipeline_class()