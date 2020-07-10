from typing import Dict, List

import numpy as np
import torch as t
import torch.nn.functional as F
from tqdm import tqdm

from unify_eval.model.mixins.classification import Classifier
from unify_eval.model.mixins.sequences.seq_input import SequenceInputModel
from unify_eval.model.types import Tensor, ListOfRawTexts, ListOfTokenizedTexts
from unify_eval.utils.label_mapper import LabelMapper
from unify_eval.utils.text_sequence import SequenceMapper, Tokenizer


class ResNetLayer(t.nn.Module):
    """
    Single pytorch ResNet layer
    """

    def __init__(self,
                 n_channels: int,
                 kernel_size: int,
                 padding: int,
                 dilation: int,
                 length: int) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.convs = t.nn.ModuleList([
            t.nn.Conv1d(in_channels=n_channels,
                        out_channels=n_channels,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        padding=padding,
                        ) for _ in range(length)])

        self.activations = t.nn.ModuleList([
            t.nn.ReLU() for _ in range(length)])
        self.bns = t.nn.ModuleList([
            t.nn.BatchNorm1d(num_features=n_channels)
        ])

    def forward(self, x: t.Tensor) -> t.Tensor:
        x_ = x
        for conv, act, bn in zip(self.convs, self.activations, self.bns):
            x_ = bn(act(conv(x_)))
        return x + x_


class ResNet(t.nn.Module):
    """
    Pytorch implementation of a ResNet model
    """

    def __init__(self,
                 resnet_layers: List[ResNetLayer],
                 clf: t.nn.Module,
                 embedding_index: Dict[str, np.ndarray] = None,
                 sequence_mapper: SequenceMapper = None) -> None:
        """
        :param resnet_layers: list of ResNetLayer instances
        :param clf: classifier module attached to ResNet encoder
        :param embedding_index: map from token to embedding
        :param sequence_mapper: SequenceMapper instance generating token index sequences
        """
        super().__init__()
        embedding_dim = embedding_index[list(embedding_index.keys())[0]].shape[-1]
        embeddings_matrix = np.zeros((len(embedding_index), embedding_dim))
        print("filling embedding matrix with predtrained embeddings")
        for word, i in tqdm(sequence_mapper.vocab.stoi.items()):
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embeddings_matrix[i] = embedding_vector
        embeddings_matrix = t.from_numpy(embeddings_matrix).float()
        self.embedding_matrix = t.nn.Embedding.from_pretrained(embeddings=embeddings_matrix)
        self.resnet_layers = t.nn.ModuleList(resnet_layers)
        self.clf = clf

    def forward(self, indices: t.Tensor) -> t.Tensor:
        embeddings: t.Tensor = self.embedding_matrix(indices).permute((0, 2, 1))
        x = t.nn.Conv1d(in_channels=self.embedding_matrix.weight.shape[-1],
                        out_channels=self.resnet_layers[0].n_channels, kernel_size=1)(embeddings)
        for l in self.resnet_layers:
            x = l(x)

        return self.clf(F.elu(x.flatten(start_dim=1)))


class ResNetModel(Classifier, SequenceInputModel):
    """
    Wrapper around pytorch implementation of a ResNet model
    """

    def __init__(self,
                 label_mapper: LabelMapper,
                 tokenizer: Tokenizer,
                 sequence_mapper: SequenceMapper,
                 resnet: ResNet,
                 max_len: int,
                 text_kw: str = "texts",
                 label_kw: str = "label_kw"):
        """
        :param label_mapper: LabelMapper instance mapping labels to their indices and vice versa
        :param tokenizer: tokenizer to use
        :param sequence_mapper: SequenceMapper instance generating token index sequences
        :param resnet: actual ResNet module
        :param max_len: maximum input length, remainder is cut off
        """
        Classifier.__init__(self, label_mapper)
        SequenceInputModel.__init__(self, tokenizer=tokenizer, sequence_mapper=sequence_mapper)
        self.resnet = resnet
        self.max_len = max_len
        self.text_kw = text_kw
        self.label_kw = label_kw
        self._opt = t.optim.Adam(params=list(self.resnet.parameters()))
        self._opt.zero_grad()
        self._xent = t.nn.CrossEntropyLoss()

    def predict_label_probabilities(self, clauses: ListOfRawTexts, **kwargs) -> Tensor:
        with t.no_grad():
            return F.softmax(self.get_logits(clauses=clauses, **kwargs)).detach().numpy()

    def get_logits(self, clauses: ListOfRawTexts, **kwargs) -> Tensor:
        indices = t.from_numpy(
            self.map_to_index_sequence(raw_texts=clauses, max_length=self.max_len)).long()
        return self.resnet.forward(indices=indices)

    def map_to_embedding_sequence(self, max_length: int,
                                  raw_texts: ListOfRawTexts = None,
                                  tokenized_texts: ListOfTokenizedTexts = None,
                                  **kwargs) -> Tensor:
        indices = self.map_to_index_sequence(tokenized_texts=tokenized_texts, raw_texts=raw_texts,
                                             max_length=max_length)
        return self.resnet.embedding_matrix(indices)

    def train(self, **kwargs) -> "ResNetModel":
        loss = self.get_loss(as_tensor=True, **kwargs)["cross_entropy"]
        loss.backward()
        self._opt.step()
        self._opt.zero_grad()
        return self

    def get_loss(self, as_tensor=False, **kwargs) -> Dict[str, Tensor]:
        loss = self._xent.forward(input=self.get_logits(**kwargs),
                                  target=t.from_numpy(self.label_mapper.map_to_indices(kwargs[self.label_kw])).long())
        if not "as_tensor":
            loss = loss.detach().item()
        return {
            "cross_entropy": loss
        }

    @classmethod
    def from_components(cls, **kwargs) -> "ResNetModel":
        return cls(**kwargs)

    def get_components(self) -> dict:
        return {
            "label_mapper": self.label_mapper,
            "sequence_mapper": self.sequence_mapper,
            "resnet": self.resnet,
            "max_len": self.max_len,
            "text_kw": self.text_kw,
            "label_kw": self.label_kw
        }

    def get_numpy_parameters(self) -> Dict[str, np.ndarray]:
        return dict((name, p.detach().numpy()) for name, p in self.resnet.named_parameters())
