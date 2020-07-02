from typing import Dict, Iterator

import numpy as np
import torch as t
from fastai.text import to_detach

from unifyeval.model.layers.layer_base import Layer
from unifyeval.model.mixins.stateful import StatefulModel
from unifyeval.model.types import Tensor


class LSTMLayer(t.nn.LSTM, Layer, StatefulModel):
    """
    Subclass of both torch.nn.LSTM, Layer and StatefulModel.
    """
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 bias: bool = True,
                 batch_first: bool = True,
                 dropout: float = 0.0,
                 bidirectional: bool = False):
        """
        :param input_size: integer defining dimensionality of input
        :param hidden_size: integer definining dimensionality of hidden state
        :param num_layers: number of LSTM layers
        :param bias: if True, bias is used in LSTM cell
        :param batch_first: if True, data is supposed to be batch-first, otherwise time-step-first
        :param dropout: float defining dropout rate during training.
        :param bidirectional: if True, bidirectional LSTMs are used.
        """
        super().__init__(input_size,
                         hidden_size,
                         num_layers,
                         bias,
                         batch_first=batch_first,
                         dropout=dropout,
                         bidirectional=bidirectional)
        self.state = self.get_default_state()

    def get_default_state(self) -> object:
        return None

    def set_state(self, state: object) -> "LSTMLayer":
        self.state = state
        return self

    def get_state(self) -> object:
        return self.state

    def push(self, **kwargs) -> dict:
        embeddings: t.Tensor = kwargs["embeddings"]
        input_lengths = kwargs["input_lengths"]
        packed = t.nn.utils.rnn.pack_padded_sequence(embeddings, input_lengths, enforce_sorted=False)

        outputs, hidden = self(input=packed, hx=self.get_state())
        # detach state
        hidden = to_detach(hidden)
        self.set_state(state=hidden)

        outputs, output_lengths = t.nn.utils.rnn.pad_packed_sequence(sequence=outputs)

        kwargs["embeddings"] = outputs
        kwargs["output_lengths"] = output_lengths
        return kwargs

    def train_mode(self) -> "LSTMLayer":
        self.train()
        return self

    def eval_mode(self) -> "LSTMLayer":
        self.eval()
        return self

    def get_components(self) -> dict:
        return dict(input_size=self.input_size,
                    hidden_size=self.hidden_size,
                    num_layers=self.num_layers,
                    bias=self.bias,
                    batch_first=self.batch_first,
                    dropout=self.dropout,
                    bidirectional=self.bidirectional,
                    state_dict=self.state_dict())

    @classmethod
    def from_components(cls, **kwargs) -> "LSTMLayer":
        state_dict = kwargs.pop("state_dict")
        layer = LSTMLayer(**kwargs)
        layer.load_state_dict(state_dict)
        return layer

    def get_numpy_parameters(self) -> Dict[str, np.ndarray]:
        return dict((name, p.detach().numpy()) for name, p in self.named_parameters())

    def get_optimizable_parameters(self) -> Iterator[Tensor]:
        return self.parameters()


class ExtractLastState(Layer):
    """
    Layer to extract the last output state of a recurrent model.
    """

    def push(self, **kwargs) -> dict:
        # simply get last output of last slice after unrolling
        outputs, output_lengths = kwargs["embeddings"][0], kwargs["output_lengths"][0]
        # outputs are time-first, so permute first two axes to get shape [batch,time,dim_hidden]
        outputs = outputs.permute((1, 0, 2))
        last_states = outputs[np.arange(outputs.shape[0]), output_lengths - 1, :]
        kwargs["text_embedding"] = last_states
        return kwargs
