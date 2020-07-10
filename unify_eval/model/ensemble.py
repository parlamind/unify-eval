from typing import List, Dict

import numpy as np
import torch as t
import torch.nn.functional as F
from keras_preprocessing import sequence
from sklearn.feature_extraction.text import HashingVectorizer

from unify_eval.model.mixins.classification import Classifier
from unify_eval.model.types import Tensor
from unify_eval.utils.label_mapper import LabelMapper


class FastText(t.nn.Module):
    """
    Pytorch implementation  of FastText.
    """
    def __init__(self, embedding_bag: t.nn.EmbeddingBag, clf: t.nn.Module) -> None:
        super().__init__()
        self.embedding_bag = embedding_bag
        self.clf = clf

    def forward(self, indices: t.Tensor) -> t.Tensor:
        return self.clf(self.embedding_bag.forward(indices))


class Ensemble(t.nn.Module):
    """
    Weighted Ensemble of different sub-models.
    """
    def __init__(self, models: List[t.nn.Module]) -> None:
        super().__init__()
        self.models = t.nn.ModuleList(models)
        self.model_weights = t.nn.Linear(in_features=len(models), out_features=1)

    def forward(self, x: t.Tensor) -> t.Tensor:
        raw_predictions = t.stack([model(x) for model in self.models], dim=-1)
        return self.model_weights(raw_predictions).view((-1, raw_predictions.shape[-2]))


class EnsembleModel(Classifier):
    """
    Text classifier that implements a linear combination over simpler models
    """
    def __init__(self,
                 label_mapper: LabelMapper,
                 ensemble: Ensemble,
                 hashing_vectorizer: HashingVectorizer,
                 max_features: int,
                 text_kw: str = "texts",
                 label_kw: str = "labels"
                 ):
        """
        :param label_mapper: LabelMapper instance mapping label name to respective index and vice versa
        :param ensemble: actual pytorch model
        :param hashing_vectorizer: hashing vectorizer generating ngram features
        :max_len: maximum number of ngram features used.
        """
        super().__init__(label_mapper)
        self.ensemble = ensemble
        self.hashing_vectorizer = hashing_vectorizer
        self.max_features = max_features
        self.text_kw = text_kw
        self.label_kw = label_kw
        self._xent = t.nn.CrossEntropyLoss()
        self._opt = t.optim.Adam(params=list(self.ensemble.parameters()))
        self._opt.zero_grad()

    def preprocess_clauses(self, clauses: List[str]) -> t.Tensor:
        """
        maps list of clauses to padded sequence of ngram indices
        """
        onehots = self.hashing_vectorizer.transform(clauses).toarray()
        sequences = np.array([np.arange(d.shape[-1])[d > 0.1] for d in onehots])
        return t.from_numpy(sequence.pad_sequences(sequences=sequences, maxlen=self.max_features)).long()

    def predict_label_probabilities(self, **kwargs) -> Tensor:
        return F.softmax(self.get_logits(**kwargs), dim=-1).detach().numpy()

    def get_logits(self, **kwargs) -> Tensor:
        indices = self.preprocess_clauses(clauses=kwargs[self.text_kw])
        return self.ensemble.forward(x=indices)

    def train(self, **kwargs) -> "EnsembleModel":
        loss = self.get_loss(as_tensor=True, **kwargs)["cross_entropy"]
        loss.backward()
        self._opt.step()
        self._opt.zero_grad()
        return self

    def get_loss(self, as_tensor: bool = False, **kwargs) -> Dict[str, Tensor]:
        y_true = t.from_numpy(self.label_mapper.map_to_indices(kwargs[self.label_kw])).long()
        y_pred = self.get_logits(**kwargs)
        loss = self._xent.forward(input=y_pred, target=y_true)
        if not as_tensor:
            loss = loss.detach().numpy()
        return {
            "cross_entropy": loss
        }

    @staticmethod
    def from_components(**kwargs) -> "EnsembleModel":
        return EnsembleModel(**kwargs)

    def get_components(self) -> dict:
        return {
            "ensemble": self.ensemble,
            "hashing_vectorizer": self.hashing_vectorizer,
            "max_len": self.max_features,
            "label_mapper": self.label_mapper,
            "text_kw":self.text_kw,
            "label_kw":self.label_kw
        }

    def get_numpy_parameters(self) -> Dict[str, np.ndarray]:
        return dict((name, p.detach().numpy()) for name, p in self.ensemble.named_parameters())
