import abc
from typing import List

import numpy as np

from unify_eval.model.deep_model import DeepModel
from unify_eval.model.types import Label
from unify_eval.model.types import Tensor
from unify_eval.utils.label_mapper import LabelMapper


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
