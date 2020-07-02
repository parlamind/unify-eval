from typing import List, Tuple, Dict

import torch as t
from fastai.text import transform
from torch.nn import CrossEntropyLoss

from unifyeval.model.deep_model import DeepModel
from unifyeval.model.layers.layer_base import Layer, LayeredModel, LayerContainer
from unifyeval.model.layers.preprocessing import SequenceMapperLayer, TokenizerLayer
from unifyeval.model.mixins.sequences.seq2seq import PytorchLanguageModel
from unifyeval.model.mixins.stateful import StatefulLayeredModel
from unifyeval.model.types import Tensor
from unifyeval.training.seq2seq.seq2seq_data import Seq2SeqData


class LayeredLanguageModel(StatefulLayeredModel, PytorchLanguageModel, DeepModel):
    """
    Class for stateful layered language models models with pytorch backends.
    """

    def __init__(self,
                 layers: List[Tuple[str, Layer]] = None,
                 layer_container: LayerContainer = None,
                 optimizer_factory=None):
        """
        :param layers: List of name - layer tuples
        :param layer_container: predefined LayerContainer instance holding list of layers
        :param optimizer_factory: callable with an argument named "params" that returns some pytorch optimizer
        """
        StatefulLayeredModel.__init__(self, layers, layer_container)
        PytorchLanguageModel.__init__(self,
                                      tokenizer=self.layers.preprocessing.layers.tokenizer,
                                      sequence_mapper=self.layers.preprocessing.layers.sequence_mapper)
        self.tail = self[1:]
        optimizer_factory = optimizer_factory if optimizer_factory is not None else lambda params: t.optim.Adam(lr=1e-4,
                                                                                                                params=params)
        self.optimizer = optimizer_factory(params=list(self.get_optimizable_parameters()))
        self.xent = CrossEntropyLoss(
            ignore_index=self.layers.preprocessing.layers.sequence_mapper.vocab.stoi[transform.PAD])

    @classmethod
    def from_layers(cls,
                    tokenizer_layer: TokenizerLayer,
                    sequence_mapper_layer: SequenceMapperLayer,
                    core_model: StatefulLayeredModel,
                    token_classifier: LayeredModel,
                    optimizer_factory=None) -> "LayeredLanguageModel":
        preprocessing = LayeredModel([("tokenizer", tokenizer_layer),
                                      ("sequence_mapper", sequence_mapper_layer),
                                      ])

        combined = LayeredLanguageModel([
            ("preprocessing", preprocessing),
            ("core_model", core_model),
            ("token_classifier", token_classifier)
        ], optimizer_factory=optimizer_factory).reset()
        return combined

    def get_loss(self, input_indices: List[List[int]], target_indices: List[List[int]], **kwargs) -> Dict[str, Tensor]:
        # indices are already preprocessed, so push only through tail of model
        logits = self.tail.push(encoded_texts=input_indices,
                                padding_value=self.layers.preprocessing.layers.sequence_mapper.vocab.stoi["xxpad"],
                                **kwargs)["logits"]
        padded_target = t.nn.utils.rnn.pad_sequence(
            sequences=[t.tensor(indices_).long() for indices_ in target_indices],
            padding_value=self.sequence_mapper.vocab.stoi["xxpad"])
        l = self.xent(input=logits.view(-1, logits.shape[-1]), target=padded_target.view(-1, ))
        return {"xent": l}

    def train(self, data: Seq2SeqData, **kwargs) -> "LayeredLanguageModel":
        for input_data, target_data in data:
            loss = self.get_loss(input_indices=input_data, target_indices=target_data, **kwargs)["xent"]
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        return self

    def predict_target_logits(self, indices: List[List[int]]) -> Tensor:
        # indices are already preprocessed, so push only through tail of model
        return self.tail.push(encoded_texts=indices,
                              padding_value=self.sequence_mapper.vocab.stoi["xxpad"])["logits"]

    def _get_cross_entropy_singlebatch(self, input_indices: List[List[int]], target_indices: List[List[int]],
                                       batch_first: bool = False, **kwargs) -> Tensor:
        return self.get_loss(input_indices=input_indices, target_indices=target_indices)["xent"]
