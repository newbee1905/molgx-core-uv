import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from .Prediction import RegressionModel

import numpy as np

class TorchRegressionModel(RegressionModel):
    """Torch-based Regression Model."""

    def __init__(self, estimator, moldata, target_property, features, scaler):
        """Constructor for TorchRegressionModel.

        Args:
                estimator (nn.Module): a PyTorch model for regression
                moldata (MolData): a molecule data management object
                target_property (str): name of target property of regression
                features (MergedFeatureSet): a feature set for regression
                scaler (object): scaling for standardizing data
        """
        super().__init__(estimator, moldata, target_property, features, scaler)
        if not isinstance(estimator, nn.Module):
            raise TypeError("estimator must be a PyTorch nn.Module instance")
        self.estimator = estimator
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.estimator.to(self.device)
        self.optimizer = None
        self.criterion = nn.MSELoss()

    def fit(self, data=None, epochs=100, lr=1e-3):
        self.optimizer = torch.optim.AdamW(self.estimator.parameters(), lr=lr)

        features = self.get_selected_data(dataframe=None if data is None else data)
        target = self.get_target()

        features_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
        target_tensor = torch.tensor(target, dtype=torch.float32).to(self.device)

        self.estimator.train()

        for epoch in range(epochs):
            self.optimizer.zero_grad()
            predictions = self.estimator(features_tensor).squeeze()
            loss = self.criterion(predictions, target_tensor)
            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

        self.set_mse(target_tensor.cpu().numpy(), predictions.detach().cpu().numpy())
        self.score = 1 - (self.mse / np.var(target))

        return predictions.detach().cpu().numpy()

    def predict(self, dataframe=None):
        self.estimator.eval()

        features = self.get_selected_data(dataframe=None if dataframe is None else dataframe)
        features_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            predictions = self.estimator(features_tensor).squeeze()

        return predictions.cpu().numpy()

    def cross_validation(self, data=None, folds=5):
        features = self.get_selected_data(dataframe=None if data is None else data)
        target = self.get_target()

        kf = KFold(n_splits=folds, shuffle=True, random_state=42)
        scores = []

        for train_idx, test_idx in kf.split(features):
            train_features = torch.tensor(features[train_idx], dtype=torch.float32).to(self.device)
            train_target = torch.tensor(target[train_idx], dtype=torch.float32).to(self.device)

            test_features = torch.tensor(features[test_idx], dtype=torch.float32).to(self.device)
            test_target = torch.tensor(target[test_idx], dtype=torch.float32).to(self.device)

            # Train the model
            self.fit(data=(train_features, train_target))

            # Predict and evaluate
            predictions = self.estimator(test_features).squeeze().detach().cpu().numpy()
            mse = mean_squared_error(test_target.cpu().numpy(), predictions)
            score = 1 - (mse / np.var(test_target.cpu().numpy()))
            scores.append(score)

        self.cv_score = (np.mean(scores), np.std(scores))
        return scores

    def predict_single_val(self, vector):
        self.estimator.eval()

        vector_tensor = torch.tensor(vector, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prediction = self.estimator(vector_tensor).squeeze().item()

        return prediction

    def save_model(self, filepath):
        torch.save(self.estimator.state_dict(), filepath)

    def load_model(self, filepath):
        self.estimator.load_state_dict(torch.load(filepath))
        self.estimator.to(self.device)

class SimpleTorchRegressionModel(nn.Module):
    """A simple fully connected regression model."""

    def __init__(self, input_size, hidden_size=64, num_hidden_layers=4, output_size=1):
        """
        Args:
                input_size (int): Number of input features.
                hidden_size (int): Number of neurons in hidden layers. Defaults to 64.
                num_hidden_layers (int): Number of hidden layers. Defaults to 2.
                output_size (int): Number of output features. Defaults to 1.
        """
        super(SimpleTorchRegressionModel, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden_layers)]
        )
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """Forward pass."""
        x = F.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        x = self.output_layer(x)
        return x
