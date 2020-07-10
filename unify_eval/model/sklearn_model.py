"""
JUST FOR PROTOTYPING
"""
from typing import Dict

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss

from unify_eval.model.deep_model import DeepModel
from unify_eval.model.mixins.classification import Classifier
from unify_eval.model.types import Tensor
from unify_eval.utils.label_mapper import LabelMapper


class SklearnModel(Classifier):
    """
    Simple linear model based on ngrams
    """

    def get_logits(self, **kwargs) -> Tensor:
        raise NotImplementedError()

    def __init__(self,
                 hashing_vectorizer: HashingVectorizer,
                 clf: SGDClassifier,
                 label_mapper: LabelMapper,
                 text_kw: str = "texts",
                 label_kw: str = "labels") -> None:
        super().__init__(label_mapper)
        self.hashing_vectorizer = hashing_vectorizer
        self.clf = clf
        self.text_kw = text_kw
        self.label_kw = label_kw

    def predict_label_probabilities(self, **kwargs) -> np.array:
        return self.clf.predict_proba(self.hashing_vectorizer.transform(kwargs[self.text_kw]))

    def train(self, **kwargs) -> "DeepModel":
        self.clf.partial_fit(X=self.hashing_vectorizer.transform(kwargs[self.text_kw]),
                             y=self.label_mapper.map_to_indices(kwargs[self.label_kw]),
                             classes=kwargs["classes"])
        return self

    def get_loss(self, **kwargs) -> dict:
        all_labels = self.label_mapper.indices
        y_true = self.label_mapper.map_to_indices(kwargs[self.label_kw])
        y_pred = self.predict_label_probabilities(clauses=kwargs[self.text_kw])
        return {
            "cross_entropy": log_loss(y_true=y_true,
                                      y_pred=y_pred,
                                      labels=all_labels)
        }

    @classmethod
    def from_components(cls, **kwargs) -> "DeepModel":
        return cls(**kwargs)

    def get_numpy_parameters(self) -> Dict[str, np.ndarray]:
        return {
            "weights": self.clf.coef_,
            "bias": self.clf.intercept_
        }

    def get_components(self) -> dict:
        return {
            "hashing_vectorizer": self.hashing_vectorizer,
            "clf": self.clf,
            "label_mapper": self.label_mapper,
            "text_kw": self.text_kw,
            "label_kw": self.label_kw
        }
