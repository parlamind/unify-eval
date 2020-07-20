import abc
from typing import List, Tuple

from unify_eval.model.deep_model import DeepModelBase
from unify_eval.model.layers.layer_base import Layer
from unify_eval.model.layers.preprocessing import SequenceMapperLayer, TokenizerLayer
from unify_eval.model.mixins.stateful import StatefulLayeredModel
from unify_eval.model.types import Tensor
from unify_eval.training.seq2seq.seq2seq_data import SequentialData
from unify_eval.utils.vocab import PAD


class EmbeddingModel(DeepModelBase):
    """
    abstract class for embedding models.
    """

    @abc.abstractmethod
    def embed(self, **kwargs) -> Tensor:
        pass


class TextEmbeddingModel(StatefulLayeredModel, EmbeddingModel):

    def __init__(self, layers: List[Tuple[str, Layer]]):
        super().__init__(layers)
        # init here is preprocessing and language model core (can be unrolled),
        # without the aggregator (reduces unrolled outputs)
        # init is initiated once here because slicing takes some time

    def push(self, **kwargs) -> dict:
        # tokenize data
        tokenized_texts = self.preprocessing.tokenizer.push(**kwargs)["tokenized_texts"]
        # print(f"len tokenized texts {len(tokenized_texts)}")

        # generate sequential data object
        sequential_data: SequentialData = SequentialData.from_texts(
            sequence_mapper=self.preprocessing.sequence_mapper,
            tokenized_texts=tokenized_texts,
            backprop_length=kwargs["embedding_backprop_length"])

        # send everything trough model
        aggregated_outputs = self.language_model_core.unroll(sequential_data=sequential_data,
                                                             unrollable_kw="sequential_data",
                                                             single_input_kw="encoded_texts",
                                                             padding_value=
                                                             self.preprocessing.sequence_mapper.vocab.token2id[PAD],
                                                             **kwargs)
        # aggregate everything
        return self.aggregator.push(**aggregated_outputs)

    def embed(self, **kwargs) -> Tensor:
        return self.push(**kwargs)["text_embedding"]

    @classmethod
    def from_layers(cls, tokenizer: TokenizerLayer,
                    sequence_mapper: SequenceMapperLayer,
                    language_model_core: StatefulLayeredModel,
                    aggregator: Layer) -> "TextEmbeddingModel":
        preprocessing = Layer([("tokenizer", tokenizer),
                               ("sequence_mapper", sequence_mapper),
                               ])
        return TextEmbeddingModel([
            ("preprocessing", preprocessing),
            ("language_model_core", language_model_core),
            ("aggregator", aggregator)
        ])
