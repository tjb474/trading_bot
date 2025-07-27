from abc import ABC, abstractmethod

class BasePipeline(ABC):
    """
    An abstract base class for all model training pipelines.
    It defines the common interface that every pipeline must implement.
    """
    def __init__(self):
        # Imports are placed here to be available to all child classes
        from common.config import config
        from common import data_manager
        from ml import feature_engineering
        self.config = config
        self.data_manager = data_manager
        self.feature_engineering = feature_engineering

    @abstractmethod
    def run(self):
        """
        This method must be implemented by each concrete pipeline.
        It should contain the full logic for loading data, training a model,
        and saving it.
        """
        raise NotImplementedError("The 'run' method must be implemented by the subclass.")