import abc
from functools import reduce
from typing import List, Tuple, Dict

import numpy as np
import torch as t
from torch.nn import CrossEntropyLoss

from unifyeval.model.deep_model import DeepModel
from unifyeval.model.layers.layer_base import Layer, LayerContainer
from unifyeval.model.mixins.embedding import TextEmbeddingModel
from unifyeval.model.mixins.stateful import StatefulLayeredModel
from unifyeval.model.types import Label
from unifyeval.model.types import Tensor
from unifyeval.utils.label_mapper import LabelMapper


class Classifier(DeepModel):
    """
    ABC for simple classifiers
    """

    def __init__(self, label_mapper: LabelMapper):
        self.label_mapper = label_mapper

    def predict_label(self,
                      junk_threshold: float = None,
                      junk_label_index: int = None,
                      **kwargs) -> np.array:
        """
        predict labels (not label indices!) for input data
        :param kwargs: input data
        :param junk_threshold: predictions with probabilities lower than this will default to junk_label_index. Ignored if None
        :param junk_label_index:  predictions with probabilities lower than junk_threshold will default this label index. Ignored if None
        :return: labels for given input data.
        """
        probs = self.predict_label_probabilities(**kwargs)
        max_indices = np.argmax(probs, axis=-1)
        if junk_threshold is not None and junk_label_index is not None:
            max_probs = probs[np.arange(max_indices.shape[0]), max_indices]
            max_indices = np.where(max_probs > junk_threshold, max_indices, junk_label_index)
        return self.label_mapper.map_to_actual_labels(max_indices)

    @abc.abstractmethod
    def predict_label_probabilities(self, **kwargs) -> Tensor:
        """
        predict probabilities for every index label given input data
        :param kwargs: input data
        :return: np.ndarray of shape [batch_size,n_labels]
        """
        pass

    @abc.abstractmethod
    def get_logits(self, **kwargs) -> Tensor:
        pass


class MessageLevelClassifier(DeepModel):
    """
    ABC for classifiers that take an entire message and predict one intent per clause
    """

    def __init__(self, label_mapper: LabelMapper):
        self.label_mapper = label_mapper

    def predict_label_array(self,
                            junk_threshold: float = None,
                            junk_label_index: int = None,
                            **kwargs) -> np.array:
        """
        Predict labels (not label indices!) for input data.
        Might contain ignore_index as label in cases where actual message contains less clauses than rows in the array
        :param kwargs: input data
        :param junk_threshold: predictions with probabilities lower than this will default to junk_label_index. Ignored if None
        :param junk_label_index:  predictions with probabilities lower than junk_threshold will default this label index. Ignored if None
        :return: np array of shape [batch_size,n_clauses], where n_clauses is the highest number of clauses per message in the given minibatch
        """
        probs = self.predict_label_probabilities(**kwargs)
        probs_flat = probs.reshape((-1, self.label_mapper.n_labels))
        max_indices = np.argmax(probs, axis=-1)
        max_indices_flat = max_indices.reshape((-1,))

        if junk_threshold is not None and junk_label_index is not None:
            max_probs = probs_flat[np.arange(max_indices_flat.shape[0]), max_indices]
            max_indices = np.where(max_probs > junk_threshold, max_indices, junk_label_index)

        return np.array([self.label_mapper.map_to_actual_labels(mi) for mi in max_indices]).reshape(max_indices.shape)

    def predict_labels(self,
                       junk_threshold: float = None,
                       junk_label_index: int = None,
                       text_kw: str = "clauses",
                       **kwargs) -> [List[List[Label]]]:
        """
        predict labels (not label indices!) for input data
        :param kwargs: input data
        :param junk_threshold: predictions with probabilities lower than this will default to junk_label_index. Ignored if None
        :param junk_label_index:  predictions with probabilities lower than junk_threshold will default this label index. Ignored if None
        :return: label for every clause for every message in a given minibatch
        """
        label_array = self.predict_label_array(junk_threshold=junk_threshold,
                                               junk_label_index=junk_label_index,
                                               **kwargs)
        return [predictions[:len(message_clauses)] for predictions, message_clauses in
                zip(label_array, kwargs[text_kw])]

    @abc.abstractmethod
    def predict_label_probabilities(self, **kwargs) -> Tensor:
        """
        predict probabilities for every index label given input data
        :param kwargs: input data
        :return: tensor of shape [batch_size,n_clauses,n_labels]
        """
        pass

    @abc.abstractmethod
    def get_logits(self, **kwargs) -> Tensor:
        """
        predict logits for every index label given input data
        :param kwargs: input data
        :return: tensor of shape [batch_size,n_clauses,n_labels]
        """
        pass


class StatefulTextClassifier(StatefulLayeredModel, Classifier):
    """
    classifier that embeds a text and then pushes it through some dedicated classifier
    """

    def predict_label_probabilities(self, **kwargs) -> Tensor:
        logits = self.get_logits(**kwargs)
        return t.softmax(logits, dim=-1).detach().numpy()

    def __init__(self,
                 layers: List[Tuple[str, Layer]] = None,
                 layer_container: LayerContainer = None,
                 optimizer_factory=None,
                 label_mapper: LabelMapper = None):
        StatefulLayeredModel.__init__(self, layers, layer_container)
        Classifier.__init__(self, label_mapper)
        optimizer_factory = optimizer_factory \
            if optimizer_factory is not None \
            else lambda params: t.optim.Adam(lr=1e-3, params=params)
        self.optimizer = optimizer_factory(params=self.get_optimizable_parameters())
        self.xent = CrossEntropyLoss()

    def get_logits(self, **kwargs) -> Tensor:
        return self.push(**kwargs)["logits"]

    def train(self, **kwargs) -> "StatefulTextClassifier":
        # sum up loss values
        loss: t.Tensor = reduce(lambda a, b: a + b, self.get_loss(as_tensor=True, **kwargs).values(), 0)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return self

    def get_loss(self, as_tensor=False, **kwargs) -> Dict[str, Tensor]:
        labels = kwargs["labels"]
        targets = t.from_numpy(self.label_mapper.map_to_indices(labels)).long()
        logits = self.get_logits(**kwargs)
        cross_entropy = self.xent(logits, targets)
        if not as_tensor:
            cross_entropy = cross_entropy.detach().item()
        return dict(cross_entropy=cross_entropy)

    @classmethod
    def from_layers(cls,
                    text_embedding_model: TextEmbeddingModel,
                    label_classifier: Layer,
                    label_mapper: LabelMapper,
                    optimizer_factory=None,
                    ) -> "StatefulTextClassifier":
        return StatefulTextClassifier([
            ("text_embedding_model", text_embedding_model),
            ("label_classifier", label_classifier)
        ],
            optimizer_factory=optimizer_factory,
            label_mapper=label_mapper)
