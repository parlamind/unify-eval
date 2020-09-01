from typing import Callable

import torch as t

from unify_eval.model.transformer_clf import MLP


class RecurrentClassifier(t.nn.Module):
    """
    Simple Recurrent classifier that encodes a single sequence,
    extracts the maximum output and feeds it to a feedforward network.
    Can be used for instance as a decoder on top of an (attention-based) encoder
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 recurrence_type: str = "lstm",
                 num_recurrent_layers: int = 1,
                 activation: Callable = t.nn.ReLU,
                 dropout: float = 0.1,
                 skip_connect: bool = False) -> None:
        """
        :param input_dim: input sequence dimensionality
        :param output_dim: output sequence dimensionality
        :param recurrence_type: type of rnn used, either "lstm", "gru" or otherwise uses a simple RNN
        """
        super().__init__()
        self.hidden_dim = input_dim
        if recurrence_type == "lstm":
            self.recurrent_layer = t.nn.LSTM(input_dim, self.hidden_dim, batch_first=True,
                                             num_layers=num_recurrent_layers, bidirectional=True, dropout=dropout)
        elif recurrence_type == "gru":
            self.recurrent_layer = t.nn.GRU(input_dim, self.hidden_dim, batch_first=True,
                                            num_layers=num_recurrent_layers, bidirectional=True, dropout=dropout)
        else:
            self.recurrent_layer = t.nn.RNN(input_dim, self.hidden_dim, batch_first=True,
                                            num_layers=num_recurrent_layers, bidirectional=True, dropout=dropout)
        self.feedforward_layer = MLP([self.hidden_dim*2, output_dim], activation=activation, dropout=dropout)
        self.skip_connect = skip_connect

    def forward(self, input: t.Tensor, attention_mask: t.Tensor = None):
        if attention_mask is None:
            output = self.recurrent_layer(input)[0]
        else:
            seq_lengths = [sum(units).item() for units in attention_mask]
            packed = t.nn.utils.rnn.pack_padded_sequence(input, seq_lengths, batch_first=True, enforce_sorted=False)
            intermediate_output = self.recurrent_layer(packed)[0]
            output = t.nn.utils.rnn.pad_packed_sequence(intermediate_output, batch_first=True)[0]

        output = output.max(axis=-2)[0]
        if self.skip_connect:
            avg_embeddings = input.mean(axis=-2)
            doubled = t.cat([avg_embeddings]*2, dim=1)
            output = output + doubled
        return self.feedforward_layer(output)
