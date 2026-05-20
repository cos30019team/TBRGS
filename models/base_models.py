"""Abstract base class for traffic prediction models."""

from abc import ABC, abstractmethod
class BaseTrafficModel(ABC):
    """Abstract base class for traffic prediction models."""

    @abstractmethod
    def train(self, X_train, y_train, X_val, y_val, config= None):
        """Train the model on the training data."""
        
        """Arg:
            X_train: Training features (e.g., historical traffic data, weather data, etc.)
            y_train: Training labels (e.g., traffic conditions)
            X_val: Validation features
            y_val: Validation labels
            config: Optional configuration parameters for training (e.g., hyperparameters, training settings, etc
            
            Returns:
            history: Training history (e.g., loss, accuracy, etc.) for analysis and visualization.
        """
        pass

    @abstractmethod
    def predict(self, X_test):
        """Predict traffic conditions for the test data."""
        pass

    @abstractmethod
    def evaluate(self, X_test, y_test):
        """Evaluate the model's performance on the test data."""
        pass
    
    @abstractmethod
    def save(self, file_path):
        """Save the trained model to a file."""
        pass
    
    @abstractmethod
    def load(self, file_path):
        """Load a trained model from a file."""
        pass
    
    @abstractmethod
    def get_name(self):
        """Get the model's name for identification and reproducibility."""
        pass