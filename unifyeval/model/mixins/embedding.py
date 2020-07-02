import abc
from typing import List, Tuple

from unifyeval.model.deep_model import DeepModelBase
from unifyeval.model.layers.layer_base import Layer, LayeredModel, LayerContainer
from unifyeval.model.layers.preprocessing import FastAITokenizerLayer, SequenceMapperLayer
from unifyeval.model.mixins.stateful import StatefulLayeredModel
from unifyeval.model.types import Tensor
from unifyeval.training.seq2seq.seq2seq_data import SequentialData


class EmbeddingModel(DeepModelBase):
    """
    abstract class for embedding models.
    """

    @abc.abstractmethod
    def embed(self, **kwargs) -> Tensor:
        pass


class TextEmbeddingModel(StatefulLayeredModel, EmbeddingModel):

    def __init__(self, layers: List[Tuple[str, Layer]] = None, layer_container: LayerContainer = None):
        super().__init__(layers, layer_container)
        # init here is preprocessing and language model core (can be unrolled),
        # without the aggregator (reduces unrolled outputs)
        # init is initiated once here because slicing takes some time

    def push(self, **kwargs) -> dict:
        # tokenize data
        tokenized_texts = self.layers.preprocessing.layers.tokenizer.push(**kwargs)["tokenized_texts"]
        # print(f"len tokenized texts {len(tokenized_texts)}")

        # generate sequential data object
        sequential_data: SequentialData = SequentialData.from_texts(
            sequence_mapper=self.layers.preprocessing.layers.sequence_mapper,
            tokenized_texts=tokenized_texts,
            backprop_length=kwargs["embedding_backprop_length"])

        # print(f"seq data shape {sequential_data.batched_input_data.shape}")

        # send everything trough model
        aggregated_outputs = self.layers.language_model_core.unroll(sequential_data=sequential_data,
                                                                    unrollable_kw="sequential_data",
                                                                    single_input_kw="encoded_texts",
                                                                    padding_value=
                                                                    self.layers.preprocessing.layers.sequence_mapper.vocab.stoi[
                                                                        "xxpad"],
                                                                    **kwargs)
        # aggregate everything
        return self.layers.aggregator.push(**aggregated_outputs)

    def embed(self, **kwargs) -> Tensor:
        return self.push(**kwargs)["text_embedding"]

    @classmethod
    def from_layers(cls, tokenizer: FastAITokenizerLayer,
                    sequence_mapper: SequenceMapperLayer,
                    language_model_core: StatefulLayeredModel,
                    aggregator: Layer) -> "TextEmbeddingModel":
        preprocessing = LayeredModel([("tokenizer", tokenizer),
                                      ("sequence_mapper", sequence_mapper),
                                      ])
        return TextEmbeddingModel([
            ("preprocessing", preprocessing),
            ("language_model_core", language_model_core),
            ("aggregator", aggregator)
        ])
