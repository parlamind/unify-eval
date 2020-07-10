from typing import List, Dict, Iterator

import numpy as np
import torch as t

from unify_eval.model.layers.layer_base import Layer
from unify_eval.model.types import Tensor


class TokenEmbeddingLayer(t.nn.Embedding, Layer):
    """
    Layer to map sequences of token indices to respective embeddings.
    """

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 encoded_texts_kw: str = "encoded_texts",
                 padding_value_kw: str = "padding_value",
                 embedding_output_kw: str = "embeddings",
                 input_lengths_kw: str = "input_lengths"):
        """
        :param num_embeddings: total number of embeddings stored
        :param embedding_dim: dimensionality of embedding space
        :param encoded_texts_kw: keyword by which to extract list of index sequences from previous layers
        :param padding_value_kw: keyword by which to extract padding value
        :param embedding_output_kw: keyword that maps to resulting embedding tensor
        :param input_lengths_kw: keyword that maps to resulting integer list of input lengths
        """
        t.nn.Embedding.__init__(self,num_embeddings, embedding_dim)
        Layer.__init__(self)
        self.encoded_texts_kw = encoded_texts_kw
        self.padding_value_kw = padding_value_kw
        self.embedding_output_kw = embedding_output_kw
        self.input_lengths_kw = input_lengths_kw

    def push(self, **kwargs) -> dict:
        encoded_texts: List[List[int]] = kwargs[self.encoded_texts_kw]
        padding_value = kwargs[self.padding_value_kw]
        padded_indices = t.nn.utils.rnn.pad_sequence(
            sequences=[t.tensor(text_indices).long() for text_indices in encoded_texts],
            padding_value=padding_value)
        embeddings = self(input=padded_indices)

        kwargs[self.embedding_output_kw] = embeddings
        kwargs[self.input_lengths_kw] = [len(indices) for indices in encoded_texts]
        return kwargs

    def get_components(self) -> dict:
        return {"num_embeddings": self.num_embeddings,
                "embedding_dim": self.embedding_dim,
                "state_dict": self.state_dict(),
                "encoded_texts_kw": self.encoded_texts_kw,
                "padding_value_kw": self.padding_value_kw,
                "embedding_output_kw": self.embedding_output_kw,
                "input_lengths_kw": self.input_lengths_kw}

    def get_numpy_parameters(self) -> Dict[str, np.ndarray]:
        return dict((name, p.detach().numpy()) for name, p in self.named_parameters())

    @classmethod
    def from_components(cls, **kwargs) -> "TokenEmbeddingLayer":
        layer = TokenEmbeddingLayer(num_embeddings=kwargs["num_embeddings"],
                                    embedding_dim=kwargs["embedding_dim"],
                                    encoded_texts_kw=kwargs["encoded_texts_kw"],
                                    padding_value_kw=kwargs["padding_value_kw"],
                                    embedding_output_kw=kwargs["embedding_output_kw"],
                                    input_lengths_kw=kwargs["input_lengths_kw"])
        layer.load_state_dict(kwargs["state_dict"])
        return layer

    def get_optimizable_parameters(self) -> Iterator[Tensor]:
        return self.parameters()
