from collections import Counter
from typing import Dict, List

import numpy as np

from unifyeval.model.mixins.classification import Classifier
from unifyeval.model.types import Tensor, Label
from unifyeval.utils.label_mapper import LabelMapper
from unifyeval.utils.other_utils import xor


class MajorityClassifier(Classifier):
    def __init__(self, label_mapper: LabelMapper,
                 target_labels: List[Label] = None,
                 most_frequent_label: Label = None):
        super().__init__(label_mapper)
        assert xor(target_labels is None, most_frequent_label is None), \
            "either target_labels or most_frequent_label has to be None"

        assert len(target_labels) > 0, "target_labels should contain at least one data point!"

        if most_frequent_label is not None:
            self.most_frequent_label = most_frequent_label
        else:
            label_counts = Counter(target_labels)
            self.most_frequent_label = target_labels[0]
            for label, count in label_counts.items():
                if count > label_counts[self.most_frequent_label]:
                    self.most_frequent_label = label

    def predict_label_probabilities(self, **kwargs) -> Tensor:
        n_datapoints = len(kwargs[kwargs["text_kw"]])
        probs = np.full(shape=(n_datapoints, self.label_mapper.n_labels),
                        fill_value=0.0)
        for i in range(n_datapoints):
            probs[i, self.label_mapper.actuallabel2index[self.most_frequent_label]] = 1.0
        return probs

    def get_logits(self, **kwargs) -> Tensor:
        raise NotImplementedError()

    def train(self, **kwargs) -> "MajorityClassifier":
        return self

    def get_loss(self, **kwargs) -> Dict[str, Tensor]:
        return dict()

    def get_components(self) -> dict:
        return dict(label_mapper=self.label_mapper,
                    most_frequent_label=self.most_frequent_label)

    def get_numpy_parameters(self) -> Dict[str, np.ndarray]:
        return dict()

    @classmethod
    def from_components(cls, **kwargs) -> "MajorityClassifier":
        return cls(**kwargs)
