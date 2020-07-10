from functools import reduce
from typing import List, Tuple, Dict

import torch as t
from torch.nn import CrossEntropyLoss

from unify_eval.model.layers.layer_base import Layer
from unify_eval.model.mixins.classification import Classifier
from unify_eval.model.mixins.embedding import TextEmbeddingModel
from unify_eval.model.mixins.stateful import StatefulLayeredModel
from unify_eval.model.types import Tensor
from unify_eval.utils.label_mapper import LabelMapper


class StatefulTextClassifier(StatefulLayeredModel, Classifier):
    """
    classifier that embeds a text and then pushes it through some dedicated classifier
    """

    def predict_label_probabilities(self, **kwargs) -> Tensor:
        logits = self.get_logits(**kwargs)
        return t.softmax(logits, dim=-1).detach().numpy()

    def __init__(self,
                 layers: List[Tuple[str, Layer]],
                 optimizer_factory=None,
                 label_mapper: LabelMapper = None):
        StatefulLayeredModel.__init__(self, layers)
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
