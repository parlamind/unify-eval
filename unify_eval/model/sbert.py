from typing import Dict, List, Union

import numpy as np
import torch as t
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from sentence_transformers import SentenceTransformer

from unify_eval.model.mixins.classification import Classifier
from unify_eval.model.types import Tensor
from unify_eval.utils.label_mapper import LabelMapper
from unify_eval.model.transformer_clf import MLP


class SbertClassifier(t.nn.Module):
    """

    """

    def __init__(self, pretrained_model_name: str, clf: MLP) -> None:
        super().__init__()
        self.encoder = SentenceTransformer(pretrained_model_name)
        self.clf = clf

    def forward(self, text: Union[str, List[str], List[int]]) -> t.Tensor:
        with t.no_grad():
            embedded = self.encoder.encode(text, convert_to_numpy=False, convert_to_tensor=True)
        with t.enable_grad():
            return self.clf(embedded)


class SbertClassificationModel(Classifier):
    """
    """

    def __init__(self, label_mapper: LabelMapper, sbert_classifier: SbertClassifier,
                 lr: float = 0.001, weight_decay: float = 0.01):
        super().__init__(label_mapper)
        self.sbert_classifier = sbert_classifier
        self.lr = lr
        self.weight_decay = weight_decay
        self._xent = CrossEntropyLoss()
        trainable_params = list(self.sbert_classifier.clf.parameters())
        self._opt = t.optim.AdamW(params=trainable_params, lr=lr, weight_decay=weight_decay)
        self._opt.zero_grad()
        self.max_len = 512

    def predict_label_probabilities(self, **kwargs) -> Tensor:
        return F.softmax(self.get_logits(**kwargs), dim=-1).detach().cpu().numpy()

    def get_logits(self, **kwargs) -> Tensor:
        return self.sbert_classifier.forward(kwargs["clauses"])

    def train(self, **kwargs) -> "SbertClassificationModel":
        loss = self.get_loss(as_tensor=True, **kwargs)["cross_entropy"]
        loss.backward()
        self._opt.step()
        self._opt.zero_grad()
        return self

    def get_loss(self, as_tensor: bool = False, **kwargs) -> Dict[str, Tensor]:
        logits = self.get_logits(**kwargs)
        loss = self._xent.forward(input=logits,
                                  target=t.from_numpy(self.label_mapper.map_to_indices(kwargs["labels"]))
                                  .long().to(self.current_device))
        if not as_tensor:
            loss = loss.detach().cpu().item()
        return {
            "cross_entropy": loss
        }

    @staticmethod
    def from_components(**kwargs) -> "SbertClassificationModel":
        return SbertClassificationModel(**kwargs)

    def get_components(self) -> dict:
        return {
            "sbert_classifier": self.sbert_classifier,
            "label_mapper": self.label_mapper,
            "lr": self.lr,
            "weight_decay": self.weight_decay
        }

    def get_numpy_parameters(self) -> Dict[str, np.ndarray]:
        return dict((n, p.detach().cpu().numpy()) for n, p in self.sbert_classifier.named_parameters())

    def to_device(self, name: str) -> "SbertClassificationModel":
        super().to_device(name)
        self.sbert_classifier.to(name)
        return self

