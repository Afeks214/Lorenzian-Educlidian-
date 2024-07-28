import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from typing import Tuple, Union, List
from Config import get_config
from utility import calculate_lorentzian_distance, calculate_kernel_regression

class LorentzianKNN(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors: int = 5, use_lorentzian: bool = True, reset_factor: float = 0.1):
        self.n_neighbors = n_neighbors
        self.use_lorentzian = use_lorentzian
        self.reset_factor = reset_factor
        self.config = get_config()

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LorentzianKNN':
        X, y = check_X_y(X, y)
        if self.config.lorentzian.use_downsampling:
            X, y = self._apply_downsampling(X, y)
        X, y = self._apply_reset_factor(X, y)
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        X = check_array(X)
        
        distances = self._calculate_distances(X)
        neighbors = self._get_neighbors(distances)
        
        predictions = []
        for neighbor_indices in neighbors:
            neighbor_labels = self.y_[neighbor_indices]
            prediction = np.argmax(np.bincount(neighbor_labels))
            predictions.append(prediction)
        
        return np.array(predictions)

    def _calculate_distances(self, X: np.ndarray) -> np.ndarray:
        if self.use_lorentzian:
            distances = np.array([[calculate_lorentzian_distance(x1, x2) for x2 in self.X_] for x1 in X])
        else:
            distances = euclidean_distances(X, self.X_)
        return distances

    def _get_neighbors(self, distances: np.ndarray) -> np.ndarray:
        return np.argsort(distances, axis=1)[:, :self.n_neighbors]

    def _apply_reset_factor(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mask = np.random.random(len(X)) > self.reset_factor
        return X[mask], y[mask]

    def _apply_downsampling(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return X[::4], y[::4]  # Select every 4th sample

class MLModel:
    def __init__(self):
        self.config = get_config()
        self.model = LorentzianKNN(
            n_neighbors=self.config.lorentzian.n_neighbors,
            use_lorentzian=not self.config.lorentzian.reduce_warping,
            reset_factor=self.config.lorentzian.reset_factor
        )

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> 'MLModel':
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        self.model.fit(X, y)
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict(X)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.values
        distances = self.model._calculate_distances(X)
        neighbors = self.model._get_neighbors(distances)
        
        probas = []
        for neighbor_indices in neighbors:
            neighbor_labels = self.model.y_[neighbor_indices]
            class_counts = np.bincount(neighbor_labels, minlength=len(self.model.classes_))
            probas.append(class_counts / len(neighbor_indices))
        
        return np.array(probas)

class KernelRegressionModel:
    def __init__(self):
        self.config = get_config()

    def fit_predict(self, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return calculate_kernel_regression(prices, self.config.kernel_regression)

class MeanReversionDetector:
    def detect(self, price: float, kernel: float, upper: float, lower: float) -> str:
        if price > upper:
            return "strong_downward"
        elif price < lower:
            return "strong_upward"
        elif price > kernel:
            return "downward"
        elif price < kernel:
            return "upward"
        else:
            return "none"

class FirstPullbackDetector:
    def detect(self, prices: List[float], predictions: List[int]) -> Tuple[bool, bool]:
        if len(prices) < 5 or len(predictions) < 5:
            return False, False

        bullish_pullback = (predictions[-1] == 1 and prices[-1] < prices[-2] and 
                            all(pred == 1 for pred in predictions[-5:-1]))
        bearish_pullback = (predictions[-1] == 0 and prices[-1] > prices[-2] and 
                            all(pred == 0 for pred in predictions[-5:-1]))

        return bullish_pullback, bearish_pullback

class DailyKernelFilter:
    def __init__(self):
        self.config = get_config()

    def apply(self, prices: List[float], timeframe: str) -> List[float]:
        if timeframe.lower() in ['d', '1d', 'daily']:
            return prices  # No filter for daily timeframe and above
        
        return calculate_kernel_regression(np.array(prices), self.config.daily_kernel_regression)[0].tolist()

def train_and_predict(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ml_model = MLModel()
    ml_model.fit(X_train, y_train)
    predictions = ml_model.predict(X_test)
    probabilities = ml_model.predict_proba(X_test)

    kernel_model = KernelRegressionModel()
    kernel, upper, lower = kernel_model.fit_predict(prices)

    mean_reversion_detector = MeanReversionDetector()
    mean_reversion_signals = [mean_reversion_detector.detect(p, k, u, l) for p, k, u, l in zip(prices, kernel, upper, lower)]

    first_pullback_detector = FirstPullbackDetector()
    bullish_pullbacks, bearish_pullbacks = zip(*[first_pullback_detector.detect(prices[max(0, i-4):i+1], predictions[max(0, i-4):i+1]) for i in range(len(prices))])

    daily_kernel_filter = DailyKernelFilter()
    filtered_prices = daily_kernel_filter.apply(prices.tolist(), X_test.index.freqstr)

    return predictions, probabilities, kernel, upper, lower, mean_reversion_signals, bullish_pullbacks, bearish_pullbacks, filtered_prices