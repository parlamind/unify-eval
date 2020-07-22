from typing import Tuple, Dict

import numpy as np
import torch as t
import torch.nn.functional as F

from unify_eval.model.mixins.adversarial import PytorchSaliencyModel
from unify_eval.model.mixins.classification import Classifier
from unify_eval.model.mixins.embedding import EmbeddingModel
from unify_eval.model.types import Tensor, ListOfRawTexts, ListOfTokenizedTexts
from unify_eval.utils.label_mapper import LabelMapper
from unify_eval.utils.text_sequence import SequenceMapper, Tokenizer


class MLP(t.nn.Module):
    """
    Simple multilayer perceptron with intermediate batch norm layers
    """

    def __init__(self, sizes: Tuple[int, int, int]):
        t.nn.Module.__init__(self)
        self.bn0 = t.nn.BatchNorm1d(sizes[0])
        self.l0 = t.nn.Linear(in_features=sizes[0], out_features=sizes[1])
        self.bn1 = t.nn.BatchNorm1d(sizes[1])
        self.l1 = t.nn.Linear(in_features=sizes[1], out_features=sizes[2])

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.l1(self.bn1(F.elu(self.l0(self.bn0(x)))))


class AverageEmbeddingClassifier(t.nn.Module):
    """
    pytorch module that averages an embedding sequence and feeds it to an MLP classifier.
    """

    def __init__(self, vocab_size: int, token_embedding_dim: int, positional_embedding_dim: int, n_classes: int):
        """
        :param vocab_size: number of vocabulary items
        :param token_embedding_dim: dimensionality of embedding space
        :param positional_embedding_dim: dimensionality of positional embeddings
        :param n_classes: number of labels to classify
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embeddings = t.nn.Embedding(num_embeddings=vocab_size, embedding_dim=token_embedding_dim)
        self.positional_embeddings = t.nn.Embedding(num_embeddings=200, embedding_dim=positional_embedding_dim)
        # self.embedding_bag = t.nn.EmbeddingBag(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.embedding_dim = positional_embedding_dim + token_embedding_dim
        self.mlp = MLP((self.embedding_dim, 128, n_classes))

    def embed_input(self, indices: t.Tensor) -> t.Tensor:
        """
        takes a batch of token index sequences and maps them to a concatenation of token and positional embeddings
        """
        token_embeddings = self.embeddings.forward(input=indices)
        positional_indices = t.from_numpy(np.array([np.arange(indices.shape[-1]) for _ in indices]))
        positional_embeddings = self.positional_embeddings.forward(input=positional_indices)
        return t.cat((token_embeddings, positional_embeddings), dim=-1)

    def embed_onehots(self, onehots: t.Tensor) -> t.Tensor:
        """
        embeds onehot sequence of token indices to a concatenation of token and positional embeddings
        """
        token_embeddings = onehots @ self.embeddings.weight.data
        positional_indices = t.from_numpy(np.array([np.arange(onehots.shape[-2]) for _ in range(onehots.shape[0])]))
        positional_embeddings = self.positional_embeddings.forward(input=positional_indices)
        return t.cat((token_embeddings, positional_embeddings), dim=-1)

    def embed_token_embeddings(self, token_embeddings: t.Tensor) -> t.Tensor:
        """
        takes embedding tokens and concatenates them with positional embeddings
        """
        positional_indices = t.from_numpy(np.array([np.arange(token_embeddings.shape[-2]) for _ in token_embeddings]))
        positional_embeddings = self.positional_embeddings.forward(input=positional_indices)
        return t.cat((token_embeddings, positional_embeddings), dim=-1)

    def classify(self, embeddings: t.Tensor) -> t.Tensor:
        """
        takes an embedded texts, averages it and feeds it to the classifier
        """
        return self.mlp(embeddings.mean(dim=-2))

    def forward(self, indices: t.Tensor) -> t.Tensor:
        """
        takes token indices, embeds them and feeds it to the classifier
        """
        return self.classify(embeddings=self.embed_input(indices=indices))


class AverageEmbeddingModel(PytorchSaliencyModel, EmbeddingModel, Classifier):
    """
    Model that embeds a token sequence, averages it and feeds it to a classifier
    """

    def __init__(self,
                 tokenizer: Tokenizer,
                 sequence_mapper: SequenceMapper,
                 average_embedding_classifier: AverageEmbeddingClassifier,
                 label_mapper: LabelMapper,
                 text_kw: str = "texts",
                 label_kw: str = "labels"):
        PytorchSaliencyModel.__init__(self,
                                      tokenizer=tokenizer,
                                      sequence_mapper=sequence_mapper)
        Classifier.__init__(self, label_mapper=label_mapper)
        self.sequence_mapper = sequence_mapper
        self.average_embedding_classifier = average_embedding_classifier
        self.text_kw = text_kw
        self.label_kw = label_kw
        self._xent = t.nn.CrossEntropyLoss()
        self._opt = t.optim.Adam(lr=1e-3, params=list(self.average_embedding_classifier.parameters()))

    def get_saliency_matrix(self, texts: ListOfRawTexts, label: int, max_length: int = None, **kwargs) -> Tensor:
        indices = t.from_numpy(self.map_to_index_sequence(raw_texts=texts, max_length=max_length)).long()
        embeddings = self.map_to_embedding_sequence(raw_texts=texts, max_length=max_length)

        loss = self.get_loss_from_embeddings(embeddings=embeddings,
                                             indices=indices,
                                             labels=np.array([label for _ in texts]))["cross_entropy"]

        gradients = self.get_gradients(tensor=loss,
                                       with_respect_to=embeddings,
                                       module=self.average_embedding_classifier)
        self.average_embedding_classifier.zero_grad()
        saliency = -t.einsum("abc,acd -> ad", gradients, embeddings.permute((0, 2, 1))).detach().cpu().numpy()
        return saliency

    def get_loss_from_embeddings(self, embeddings: Tensor, **kwargs) -> Dict[str, Tensor]:
        logits = self.average_embedding_classifier.classify(embeddings=embeddings)
        mapped_labels = t.from_numpy(self.label_mapper.map_to_indices(kwargs[self.label_kw])).long()
        loss = self._xent.forward(input=logits, target=mapped_labels)
        return {"cross_entropy": loss}

    def get_loss_from_onehots(self, onehots: Tensor, **kwargs) -> Dict[str, Tensor]:
        mapped_labels = t.from_numpy(self.label_mapper.map_to_indices(kwargs[self.label_kw])).long()
        embeddings = self.average_embedding_classifier.embed_onehots(onehots=onehots)
        logits = self.average_embedding_classifier.classify(embeddings=embeddings)
        loss = self._xent.forward(input=logits, target=mapped_labels)
        return {"cross_entropy": loss}

    def predict_label_probabilities(self, **kwargs) -> Tensor:
        with t.no_grad():
            logits = self.get_logits(**kwargs)
            return F.softmax(logits, dim=-1).detach().numpy()

    def get_logits(self, **kwargs) -> Tensor:
        tokenized_text = self.tokenizer.tokenize_all(kwargs[self.text_kw])
        indices = t.from_numpy(self.map_to_index_sequence(tokenized_texts=tokenized_text,
                                                          max_length=kwargs["max_length"]))\
            .long()\
            .to(self.current_device)
        logits = self.average_embedding_classifier(indices=indices)
        return logits

    def map_to_embedding_sequence(self,
                                  max_length: int,
                                  raw_texts: ListOfRawTexts = None,
                                  tokenized_texts: ListOfTokenizedTexts = None) -> Tensor:
        tokenized_texts = tokenized_texts if tokenized_texts else self.tokenizer.tokenize_all(raw_texts)
        indices = t.from_numpy(np.array(self.sequence_mapper.encode_texts(tokenized_texts=tokenized_texts,
                                                                          length=max_length))).long()
        return self.average_embedding_classifier.embed_input(indices=indices)

    def train(self, **kwargs) -> "AverageEmbeddingModel":
        self.average_embedding_classifier.train()

        loss = self.get_loss(**kwargs)["cross_entropy"]
        loss.backward()
        self._opt.step()
        self._opt.zero_grad()
        return self

    def get_loss(self, **kwargs) -> Dict[str, Tensor]:
        logits = self.get_logits(**kwargs)
        mapped_labels = t.from_numpy(self.label_mapper.map_to_indices(kwargs[self.label_kw])).long()
        loss = self._xent.forward(input=logits, target=mapped_labels)
        return {"cross_entropy": loss}

    @classmethod
    def from_components(cls, **kwargs) -> "AverageEmbeddingModel":
        return cls(**kwargs)

    def get_components(self) -> dict:
        return {
            "tokenizer": self.tokenizer,
            "sequence_mapper": self.sequence_mapper,
            "average_embedding_classifier": self.average_embedding_classifier,
            "label_mapper": self.label_mapper,
            "text_kw": self.text_kw,
            "label_kw": self.label_kw
        }

    def get_numpy_parameters(self) -> Dict[str, np.ndarray]:
        return dict((name, p.detach().cpu().numpy())
                    for name, p in self.average_embedding_classifier.named_parameters())

    def embed(self, **kwargs) -> np.ndarray:
        """
        embeds input to classifier logits
        """
        with t.no_grad():
            return self.get_logits(**kwargs).detach().cpu().numpy()

    def train_mode(self) -> "AverageEmbeddingModel":
        self.average_embedding_classifier.train()
        return self

    def eval_mode(self) -> "AverageEmbeddingModel":
        self.average_embedding_classifier.eval()
        return self

    def get_module(self) -> t.nn.Module:
        return self.average_embedding_classifier

    def to_device(self, name: str) -> "AverageEmbeddingModel":
        super().to_device(name)
        self.average_embedding_classifier.to(name)
        return self


