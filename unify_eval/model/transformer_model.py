from typing import Dict, List

import numpy as np
import torch as t
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedTokenizer

from unify_eval.model.mixins.classification import Classifier
from unify_eval.model.types import Tensor
from unify_eval.utils.label_mapper import LabelMapper


class TransformerClassifier(t.nn.Module):
    def __init__(self, encoder: t.nn.Module, clf: t.nn.Module, finetuning=False, clf_architecture="mlp") -> None:
        super().__init__()
        self.encoder = encoder
        self.clf = clf
        self.finetuning = finetuning
        self.clf_architecture = clf_architecture

    def forward_encoder(self, token_indices: t.Tensor, attention_mask: t.Tensor,
                        token_type_ids: t.Tensor = None) -> t.Tensor:
        return self.encoder(token_indices, attention_mask=attention_mask)[0] if token_type_ids is None \
            else self.encoder(token_indices, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]

    def forward_clf(self, embedded: t.Tensor, attention_mask: t.Tensor) -> t.Tensor:
        if self.clf_architecture == "attention":
            attention_mask_mod = attention_mask.float().unsqueeze(1)
            attention_mask_mod[attention_mask_mod == 0] = -1000
            attention_mask_mod = attention_mask_mod.repeat(1, attention_mask_mod.shape[-1], 1)
            return self.clf(embedded, attention_mask_mod)
        elif self.clf_architecture == "mlp":
            return self.clf(embedded.mean(axis=-2))
        else:
            return self.clf(embedded, attention_mask)

    def forward(self, token_indices: t.Tensor, attention_mask: t.Tensor, token_type_ids: t.Tensor = None) -> t.Tensor:
        if not self.finetuning:
            with t.no_grad():
                embedded = self.forward_encoder(token_indices, attention_mask, token_type_ids)
            with t.enable_grad():
                return self.forward_clf(embedded, attention_mask)
        else:
            embedded = self.forward_encoder(token_indices, attention_mask, token_type_ids)
            return self.forward_clf(embedded, attention_mask)


class TransformerClassificationModel(Classifier):

    def __init__(self, label_mapper: LabelMapper, transformer_classifier: TransformerClassifier,
                 tokenizer: PreTrainedTokenizer, lr: float = 0.00075):
        super().__init__(label_mapper)
        self.transformer_classifier = transformer_classifier
        self.tokenizer = tokenizer
        self._xent = CrossEntropyLoss()
        trainable_params = list(self.transformer_classifier.clf.parameters())
        if self.transformer_classifier.finetuning:
            trainable_params = list(self.transformer_classifier.encoder.parameters()) + trainable_params
        self._opt = t.optim.Adam(params=trainable_params, lr=lr)
        self._opt.zero_grad()
        self.max_len = 512

    def preprocess(self, texts: List[str]) -> Dict[str, t.Tensor]:
        tokenized_texts = [self.tokenizer.tokenize(text)[:self.max_len] for text in texts]

        # Convert token to vocabulary indices
        max_len_found = max([len(text) for text in tokenized_texts])
        indexed_texts = [self.tokenizer.convert_tokens_to_ids(text) + (max_len_found - len(text)) * [0] for text in
                         tokenized_texts]
        attention_mask = [[1 if token != 0 else 0 for token in text] + (max_len_found - len(text)) * [0] for text in
                          tokenized_texts]

        # Convert inputs to PyTorch tensors
        token_indices = t.tensor(indexed_texts).to(self.current_device)
        attention_mask = t.tensor(attention_mask).to(self.current_device)

        return {
            "token_indices": token_indices,
            "attention_mask": attention_mask
        }

    def predict_label_probabilities(self, **kwargs) -> Tensor:
        return F.softmax(self.get_logits(**kwargs), dim=-1).detach().cpu().numpy()

    def get_logits(self, **kwargs) -> Tensor:
        return self.transformer_classifier.forward(**self.preprocess(texts=kwargs["clauses"]))

    def train(self, **kwargs) -> "TransformerClassificationModel":
        loss = self.get_loss(as_tensor=True, **kwargs)["cross_entropy"]
        loss.backward()
        self._opt.step()
        self._opt.zero_grad()
        return self

    def get_loss(self, as_tensor: bool = False, **kwargs) -> Dict[str, Tensor]:
        logits = self.get_logits(**kwargs)
        loss = self._xent.forward(input=logits,
                                  target=t.from_numpy(self.label_mapper.map_to_indices(kwargs["labels"]))
                                  .long()
                                  .to(self.current_device))
        if not as_tensor:
            loss = loss.detach().cpu().item()
        return {
            "cross_entropy": loss
        }

    @staticmethod
    def from_components(**kwargs) -> "TransformerClassificationModel":
        return TransformerClassificationModel(**kwargs)

    def get_components(self) -> dict:
        return {
            "transformer_classifier": self.transformer_classifier,
            "tokenizer": self.tokenizer,

        }

    def get_numpy_parameters(self) -> Dict[str, np.ndarray]:
        return dict((n, p.detach().cpu().numpy()) for n, p in self.transformer_classifier.named_parameters())

    def to_device(self, name: str) -> "TransformerClassificationModel":
        super().to_device(name)
        self.transformer_classifier.to(name)
        return self


class BertClassificationModel(TransformerClassificationModel):

    def __init__(self, label_mapper: LabelMapper, transformer_classifier: TransformerClassifier,
                 tokenizer: PreTrainedTokenizer, lr: float = 0.00075, distilling=False):
        TransformerClassificationModel.__init__(self, label_mapper, transformer_classifier, tokenizer, lr)
        self.distilling = distilling

    def preprocess(self, texts: List[str]) -> Dict[str, t.Tensor]:
        texts = [f"[CLS] {text} [SEP]" for text in texts]
        # text = "[CLS] When does my package arrive ? [SEP]"
        tokenized_texts = [self.tokenizer.tokenize(text)[:self.max_len] for text in texts]

        max_len_found = max([len(text) for text in tokenized_texts])
        indexed_texts = [self.tokenizer.convert_tokens_to_ids(text) + (max_len_found - len(text)) * [0] for text in
                         tokenized_texts]
        attention_mask = [[1 if token != 0 else 0 for token in text] + (max_len_found - len(text)) * [0] for text in
                          tokenized_texts]

        # Convert inputs to PyTorch tensors
        token_indices = t.tensor(indexed_texts)
        attention_mask = t.tensor(attention_mask)

        tensor_dict = {
            "token_indices": token_indices.to(self.current_device),
            "attention_mask": attention_mask.to(self.current_device)
        }

        if not self.distilling:
            segments_ids = [[0] * len(text) for text in indexed_texts]
            token_type_ids = t.tensor(segments_ids)
            tensor_dict["token_type_ids"] = token_type_ids

        return tensor_dict


class RobertaClassificationModel(TransformerClassificationModel):

    def __init__(self, label_mapper: LabelMapper, transformer_classifier: TransformerClassifier,
                 tokenizer: PreTrainedTokenizer, lr: float = 0.00075):
        TransformerClassificationModel.__init__(self, label_mapper, transformer_classifier, tokenizer, lr)

    def preprocess(self, texts: List[str]) -> Dict[str, t.Tensor]:
        texts = [f"<s> {text} </s>" for text in texts]
        # text = "[CLS] When does my package arrive ? [SEP]"
        tokenized_texts = [self.tokenizer.tokenize(text)[:self.max_len] for text in texts]

        max_len_found = max([len(text) for text in tokenized_texts])
        indexed_texts = [self.tokenizer.convert_tokens_to_ids(text) + (max_len_found - len(text)) * [0] for text in
                         tokenized_texts]
        attention_mask = [[1 if token != 0 else 0 for token in text] + (max_len_found - len(text)) * [0] for text in
                          tokenized_texts]
        # Convert inputs to PyTorch tensors
        token_indices = t.tensor(indexed_texts)
        attention_mask = t.tensor(attention_mask)
        return {
            "token_indices": token_indices.to(self.current_device),
            "attention_mask": attention_mask.to(self.current_device)
        }
