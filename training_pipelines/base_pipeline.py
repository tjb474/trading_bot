from abc import ABC, abstractmethod
from common.config import config
from common import data_manager
from ml import feature_engineering

class BasePipeline(ABC):
    """
    Abstract base class for ML training pipelines.
    Defines common functionality and interface that all pipelines must implement.
    """
    def __init__(self):
        self.config = config
        self.data_manager = data_manager
        self.feature_engineering = feature_engineering

    @abstractmethod
    def run(self):
        """
        Execute the training pipeline.
        This method must be implemented by all concrete pipeline classes.
        """
        pass