import numpy as np
import pandas as pd
from TfELM.Models.KELMModel import KELMModel
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

from alzheption.extractor import AlzheptionExtractor
from alzheption.utils import custom_cross_val_score


class AlzheptionClassificator:
    def __init__(
            self,
            extractor: AlzheptionExtractor,
            hyperparameter: list[dict],
            n_splits=10, 
            n_repeats=10,
        ):
        self.extractor = extractor
        self.hyperparameter = hyperparameter
        self.n_splits = n_splits
        self.n_repeats = n_repeats

        self._x_train: np.ndarray | None = None
        self._x_test: np.ndarray | None = None
        self._y_train: np.ndarray | None = None
        self._y_test: np.ndarray | None = None
        self._x: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self._df_evaluation: pd.DataFrame | None = None

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

    @property
    def df_evaluation(self) -> pd.DataFrame:
        if self._df_evaluation is None:
            self.evaluate_with_cross_validation()
        return self._df_evaluation
    
    def evaluate_with_cross_validation(self, n_components: int | None = None):
        if n_components is None:
            n_components = min(self.x.shape)
        
        pca = PCA(n_components=n_components)
        x = pca.fit_transform(self.x)

        # Cross-validation for KELM
        data = []
        max_score = 0
        for param in tqdm(self.hyperparameter, desc="Evaluation"):
            scores = custom_cross_val_score(
                model=param.get("model"), 
                X=x, 
                y=self.y, 
                n_splits=self.n_splits, 
                n_repeats=self.n_repeats, 
                scoring=accuracy_score, 
            )
            score = np.mean(scores)
            data.append({
                **{key: val for key, val in param.items() if isinstance(val, (str, int, float))},
                "score_list": scores,
                "score_mean": score,
            })
            if max_score < score:
                max_score = score
        
        self._df_evaluation = pd.DataFrame(data)
        return data
