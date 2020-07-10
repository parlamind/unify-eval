from typing import Dict, List

import numpy as np
from keras import utils
from keras.engine import Layer
from keras.layers import Embedding
from keras.models import Sequential
from keras.preprocessing import text, sequence

from unify_eval.model.mixins.classification import DeepModel, Classifier
from unify_eval.model.types import Tensor
from unify_eval.utils.label_mapper import LabelMapper


class KerasModel(Classifier):
    """
    Wrapper around a keras classifier model.
    """

    def __init__(self,
                 tokenizer: text.Tokenizer,
                 keras_model: Sequential,
                 label_mapper: LabelMapper,
                 maxlen: int,
                 text_kw: str = "texts",
                 label_kw: str = "labels"):
        """
        :param tokenizer: tokenizer to use
        :param keras_model: actual keras model
        :param label_mapper: label mapper instance that maps label indices to label names and vice versa
        :param maxlen: maximum input length (remainder is ignored)
        :param text_kw: keyword by which to extract text input
        :param label_kw: keyword by which to extract label input
        """
        super().__init__(label_mapper)
        self.keras_model = keras_model
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.text_kw = text_kw
        self.label_kw = label_kw
        self.loss = {}

    def preprocess_texts(self, texts) -> np.ndarray:
        """
        map texts to padded index sequences
        """
        sequences = self.tokenizer.texts_to_sequences([str(text) for text in texts])
        x = sequence.pad_sequences(sequences=sequences, maxlen=self.maxlen)

        return x

    def preprocess_labels(self, labels) -> np.ndarray:
        """
        map labels to onehot indices
        """
        y = self.label_mapper.map_to_indices(labels)

        y = utils.to_categorical(y, self.label_mapper.n_labels)
        return y

    def predict_label_probabilities(self, **kwargs) -> np.array:
        x_test = self.preprocess_texts(texts=kwargs[self.text_kw])

        return self.keras_model.predict(x_test)

    def train(self, **kwargs) -> "DeepModel":
        x_train = self.preprocess_texts(kwargs[self.text_kw])
        y_train = self.preprocess_labels(kwargs[self.label_kw])

        # train_on_batch?

        history = self.keras_model.fit(x_train, y_train,
                                       batch_size=kwargs["batch_size"],
                                       epochs=kwargs["epochs"],
                                       verbose=kwargs["verbose"])

        self.loss = history.history
        return self

    def get_loss(self, **kwargs) -> dict:
        return self.loss

    @classmethod
    def from_components(cls, **kwargs) -> "DeepModel":
        return cls(**kwargs)

    def get_numpy_parameters(self) -> Dict[str, np.ndarray]:
        return {
        }

    def get_components(self) -> dict:
        return {
            "keras_model": self.keras_model,
            "label_mapper": self.label_mapper,
            "tokenizer": self.tokenizer,
            "maxlen": self.maxlen,
            "text_kw": self.text_kw,
            "label_kw": self.label_kw
        }

    def get_logits(self, **kwargs) -> Tensor:
        pass

    @staticmethod
    def pretrained_keras_model(
            tokenizer: text.Tokenizer,
            keras_layers: List[Layer],
            label_mapper: LabelMapper,
            embedding_dim: int,
            embedding_index: Dict[str, np.ndarray],
            maxlen: int,
            text_kw: str = "texts",
            label_kw: str = "labels") -> "KerasModel":
        """
        :param tokenizer: tokenizer to use
        :param keras_layers: list of layers to concatenate into single model
        :param label_mapper: label mapper instance that maps label indices to label names and vice versa
        :param embedding_dim: embedding dimensionality
        :param embedding_index: map from token to embedding
        :param maxlen: maximum input length (remainder is ignored)
        :param text_kw: keyword by which to extract text input
        :param label_kw: keyword by which to extract label input
        """

        embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, embedding_dim))
        for word, i in tokenizer.word_index.items():
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

        embedding_layer = Embedding(len(tokenizer.word_index) + 1,
                                    embedding_dim,
                                    weights=[embedding_matrix],
                                    input_length=maxlen,
                                    trainable=False)
        keras_model = Sequential([
            embedding_layer,
            *keras_layers])

        keras_model.compile(loss='categorical_crossentropy',
                            optimizer='adam',
                            metrics=['categorical_crossentropy'])

        return KerasModel(tokenizer=tokenizer,
                          keras_model=keras_model,
                          label_mapper=label_mapper,
                          maxlen=maxlen,
                          text_kw=text_kw,
                          label_kw=label_kw)
