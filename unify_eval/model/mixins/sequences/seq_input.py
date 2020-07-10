import abc

import numpy as np

from unify_eval.model.deep_model import DeepModel
from unify_eval.model.types import Tensor, ListOfRawTexts, ListOfTokenizedTexts
from unify_eval.utils.text_sequence import SequenceMapper, Tokenizer


class SequenceInputModel(DeepModel):
    """
    Class for models that expect sequential input
    """

    def __init__(self, tokenizer: Tokenizer, sequence_mapper: SequenceMapper):
        self.tokenizer = tokenizer
        self.sequence_mapper = sequence_mapper

    def map_to_index_sequence(self,
                              max_length: int,
                              raw_texts: ListOfRawTexts = None,
                              tokenized_texts: ListOfTokenizedTexts = None,
                              ) -> Tensor:
        """
        maps texts to a matrix of shape [minibatch_size,max_length]
        :param raw_texts: raw texts to encode
        :param tokenized_texts: texts to encode
        :param max_length: max length of texts. smaller texts are padded, longer ones cut off
        :return: matrix of shape [minibatch_size,max_length]
        """

        if raw_texts is not None:
            tokenized_texts = self.tokenizer.tokenize_all(raw_texts)
        return np.array(self.sequence_mapper.encode_texts(tokenized_texts=tokenized_texts, length=max_length))

    def map_to_onehot_sequence(self,
                               max_length: int,
                               raw_texts: ListOfRawTexts = None,
                               tokenized_texts: ListOfTokenizedTexts = None
                               ) -> Tensor:
        """
        maps texts to a 3d tensor of shape [minibatch_size,max_length,vocab_size]
        :param raw_texts: raw texts to encode
        :param tokenized_texts: texts to encode
        :param max_length: max length of texts. smaller texts are padded, longer ones cut off
        :return: tensor of shape [minibatch_size,max_length,vocab_size]
        """
        if raw_texts is not None:
            tokenized_texts = self.tokenizer.tokenize_all(raw_texts)
        return self.sequence_mapper.encode_texts_onehots(tokenized_texts=tokenized_texts, length=max_length)

    @abc.abstractmethod
    def map_to_embedding_sequence(self,
                                  max_length: int,
                                  raw_texts: ListOfRawTexts = None,
                                  tokenized_texts: ListOfTokenizedTexts = None) -> Tensor:
        """
        maps texts to a 3d tensor of shape [minibatch_size,max_length,embedding_dim]
        :param raw_texts: raw texts to encode
        :param tokenized_texts: texts to encode
        :param max_length: max length of texts. smaller texts are padded, longer ones cut off
        :return: matrix of shape [minibatch_size,max_length,embedding_dim]
        """
        pass
