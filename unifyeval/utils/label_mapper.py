from typing import Union, List

import numpy as np

from unifyeval.model.types import Label


class LabelMapper(object):
    """
    class that maps some symbolic output labels to indices in a probability vector
    """

    def __init__(self, Y=None, actuallabel2index: dict = None, ignore_index: int = None, unknown: Label = None):
        assert Y is not None or actuallabel2index is not None
        assert Y is None or actuallabel2index is None
        self.unknown = unknown
        if Y is not None:
            self.labels = sorted(list(set(np.unique(Y))))
            self.n_labels = len(self.labels)
            self.all_indices = np.arange(self.n_labels)

            self.actuallabel2index = dict((label, i) for i, label in enumerate(self.labels))
        else:
            self.actuallabel2index = actuallabel2index
            self.n_labels = len(self.actuallabel2index)
            self.labels = [0] * self.n_labels
            for label, i in self.actuallabel2index.items():
                self.labels[i] = label
            self.all_indices = np.arange(self.n_labels)
        self.index2actuallabel = dict((i, label) for i, label in enumerate(self.labels))
        self.label_array = np.array([self.index2actuallabel[index] for index in self.all_indices])

        self.ignore_index = ignore_index

    def _map_to_index(self, label):
        if self.ignore_index:
            if label == self.ignore_index:
                return self.ignore_index
            if self.unknown is not None:
                if label not in self.actuallabel2index:
                    return self.actuallabel2index[self.unknown]
            return self.actuallabel2index[label]
        else:

            if self.unknown is not None:
                if label not in self.actuallabel2index:
                    return self.actuallabel2index[self.unknown]
            return self.actuallabel2index[label]

    def _map_to_label(self, index: int):
        if self.ignore_index:
            return self.index2actuallabel[index] if index != self.ignore_index else self.ignore_index
        else:
            return self.index2actuallabel[index]

    def map_to_indices(self, labels: Union[np.ndarray, List[Union[str, int]]]) -> np.ndarray:
        return np.array([self._map_to_index(label=label) for label in labels])

    def map_to_actual_labels(self, indices: Union[np.ndarray, List[Union[str, int]]]) -> np.ndarray:
        return np.array([self._map_to_label(index=index) for index in indices if index != self.ignore_index])
