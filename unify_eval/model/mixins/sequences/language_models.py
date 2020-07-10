from typing import List, Tuple, Dict, Iterator

import torch as t
from fastai.text import transform, PAD
from torch.nn import CrossEntropyLoss

from unify_eval.model.deep_model import DeepModel
from unify_eval.model.layers.layer_base import Layer
from unify_eval.model.layers.preprocessing import SequenceMapperLayer, TokenizerLayer
from unify_eval.model.mixins.sequences.seq2seq import PytorchLanguageModel
from unify_eval.model.mixins.stateful import StatefulLayeredModel
from unify_eval.model.types import Tensor
from unify_eval.training.seq2seq.seq2seq_data import Seq2SeqData


class LayeredLanguageModel(StatefulLayeredModel, PytorchLanguageModel):
    """
    Class for stateful layered language models models with pytorch backends.
    """

    def __init__(self,
                 sub_layers: List[Tuple[str, Layer]],
                 optimizer_factory=None):
        """
        :param sub_layers: List of name - layer tuples
        :param optimizer_factory: callable with an argument named "params" that returns some pytorch optimizer
        """
        StatefulLayeredModel.__init__(self, sub_layers)
        PytorchLanguageModel.__init__(self,
                                      tokenizer=self.preprocessing.tokenizer,
                                      sequence_mapper=self.preprocessing.sequence_mapper)
        self.tail = self[1:]
        optimizer_factory = optimizer_factory if optimizer_factory is not None else lambda params: t.optim.Adam(lr=1e-4,
                                                                                                                params=params)
        self.optimizer = optimizer_factory(params=list(self.get_optimizable_parameters()))
        self.xent = CrossEntropyLoss(
            ignore_index=self.preprocessing.sequence_mapper.vocab.stoi[transform.PAD])

    @classmethod
    def from_layers(cls,
                    tokenizer_layer: TokenizerLayer,
                    sequence_mapper_layer: SequenceMapperLayer,
                    core_model: StatefulLayeredModel,
                    token_classifier: Layer,
                    optimizer_factory=None) -> "LayeredLanguageModel":
        preprocessing = Layer() \
                        + ("tokenizer", tokenizer_layer) \
                        + ("sequence_mapper", sequence_mapper_layer)

        combined = LayeredLanguageModel([
            ("preprocessing", preprocessing),
            ("core_model", core_model),
            ("token_classifier", token_classifier)
        ], optimizer_factory=optimizer_factory).reset()
        return combined

    def get_loss(self, input_indices: List[List[int]], target_indices: List[List[int]], **kwargs) -> Dict[str, Tensor]:
        # indices are already preprocessed, so push only through tail of model
        logits = self.tail.push(encoded_texts=input_indices,
                                padding_value=self.preprocessing.sequence_mapper.vocab.stoi[PAD],
                                **kwargs)["logits"]
        padded_target = t.nn.utils.rnn.pad_sequence(
            sequences=[t.tensor(indices_).long() for indices_ in target_indices],
            padding_value=self.sequence_mapper.vocab.stoi[PAD])
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
