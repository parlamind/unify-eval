from dataclasses import dataclass
from typing import List

import numpy as np
from fastai.text import transform

from unifyeval.utils.iter_utils import chunking, lazy_chunking
from unifyeval.utils.text_sequence import SequenceMapper


@dataclass(frozen=True)
class SequentialData:
    """
    class to store stateful sequential input data, e.g. for stateful embedding models
    """
    batched_input_data: np.ndarray

    def __iter__(self):
        for input_minibatch in self.batched_input_data:
            yield input_minibatch

    @staticmethod
    def from_texts(sequence_mapper: SequenceMapper,
                   tokenized_texts: List[List[str]],
                   backprop_length: int) -> "SequentialData":
        """
        WIP!!!!
        :param sequence_mapper:
        :param tokenized_texts:
        :param backprop_length:
        :return:
        """

        minibatch_size = len(tokenized_texts)
        encoded_texts = sequence_mapper.encode_texts(tokenized_texts=tokenized_texts)

        # flattened_tokens = [token for text in encoded_texts for token in text]
        max_len = max([len(text) for text in encoded_texts])

        # initialize index array with padding value
        n_minibatches = int(np.ceil(max_len / backprop_length))
        indices = np.full(shape=(n_minibatches, minibatch_size, backprop_length),
                          fill_value=sequence_mapper.vocab.stoi[transform.PAD])

        for i_text, text in enumerate(encoded_texts):
            for i_chunk, chunk in enumerate(lazy_chunking(text, chunk_size=backprop_length)):
                # print(chunk)
                for i_token, token_index in enumerate(chunk):
                    # print(token_index)
                    indices[i_chunk, i_text, i_token] = token_index

        # fill it! yay

        # print(f"indices shape {indices.shape} max len {max_len} sum {indices.sum()}")
        # print(indices)

        return SequentialData(batched_input_data=indices)


@dataclass(frozen=True)
class Seq2SeqData:
    """
    class to store stateful sequence-to-sequence model data. consists of input and target data.
    each minibatch is supposed to be the temporal continuation of the previous one
    """
    input_data: np.ndarray
    target_data: np.ndarray

    def __iter__(self):
        for input_minibatch, target_minibatch in zip(self.input_data, self.target_data):
            yield input_minibatch, target_minibatch

    @staticmethod
    def generate_stateful_lm_data(sequence_mapper: SequenceMapper,
                                  tokenized_texts: List[List[str]],
                                  minibatch_size: int, backprop_length: int) -> "Seq2SeqData":
        """
        generate training data of concatenated texts so that the first item of batch n is the continuation of the first item of batch n-1 etc.
        :param sequence_mapper: sequence mapper to use
        :param tokenized_texts: tokenized texts to use
        :param minibatch_size:
        :param backprop_length: backpropagation through time
        :return: np.ndarray of shape [n_chunks,minibatch_size, backprop_length] containing token indices
        """

        text_length = backprop_length + 1

        encoded_texts = sequence_mapper.encode_texts(tokenized_texts=tokenized_texts)

        flattened_tokens = [token for text in encoded_texts for token in text]
        n_tokens = len(flattened_tokens)

        # initialize index array with padding value
        n_minibatches = n_tokens // (minibatch_size * text_length)
        indices = np.full(shape=(n_minibatches, minibatch_size, text_length),
                          fill_value=sequence_mapper.vocab.stoi[transform.PAD])

        # fill it! yay

        chunks = chunking(flattened_tokens, chunk_size=text_length)

        for i_minibatch_entry in range(minibatch_size):
            for i_minibatch in range(n_minibatches):
                entry_tokens = next(chunks)
                # print(f"n_tokens {n_tokens} n_minibatches {n_minibatches} i_minibatch {i_minibatch} i_minibatch_entry {i_minibatch_entry} tokens {sm.decode_texts([entry_tokens])}")
                indices[i_minibatch, i_minibatch_entry] = entry_tokens

        input_indices = []
        target_indices = []
        for batch in indices:
            input_indices.append([token_indices[:-1] for token_indices in batch])
            target_indices.append([token_indices[1:] for token_indices in batch])
        return Seq2SeqData(input_data=np.array(input_indices),
                           target_data=np.array(target_indices))
