import numpy as np
from TfELM.Models.KELMModel import KELMModel

from alzheption.src.alzheption.extractor import AlzheptionExtractor



class AlzheptionClassificator:
    def __init__(
            self,
            extractor: AlzheptionExtractor,
            n_neurons=512, 
            n_splits=10, 
            n_repeats=10,
        ):
        self.extractor = extractor
        self.n_neurons = n_neurons
        self.n_splits = n_splits
        self.n_repeats = n_repeats

        self._x_train: np.ndarray | None = None
        self._x_test: np.ndarray | None = None
        self._y_train: np.ndarray | None = None
        self._y_test: np.ndarray | None = None
        self._x: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self._model: KELMModel | None = None

    @property
    def x_train(self) -> np.ndarray:
        if self._x_train is None:
            self._x_train = self.extractor.train_features
        return self._x_train

    @property
    def x_test(self) -> np.ndarray:
        if self._x_test is None:
            self._x_test = self.extractor.test_features
        return self._x_test

    @property
    def y_train(self) -> np.ndarray:
        if self._y_train is None:
            self._y_train = self.extractor.train_labels
        return self._y_train

    @property
    def y_test(self) -> np.ndarray:
        if self._y_test is None:
            self._y_test = self.extractor.test_labels
        return self._y_test

    @property
    def x(self) -> np.ndarray:
        if self._x is None:
            self._x = np.concatenate((self.x_train, self.x_test), axis=0)
        return self._x

    @property
    def y(self) -> np.ndarray:
        if self._y is None:
            self._y = np.concatenate((self.y_train, self.y_test), axis=0)
        return self._y
