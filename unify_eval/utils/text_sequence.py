import abc
from abc import ABC
from typing import List

import fastai.text.transform as transform
import numpy as np
from regex import regex

from unify_eval.utils.corpus import Corpus
from unify_eval.utils.iter_utils import chunking
from unify_eval.model.types import Tensor, ListOfRawTexts, ListOfTokenizedTexts


def add_bos(tokens: List[str]) -> List[str]:
    return [transform.BOS] + tokens


class Tokenizer(ABC):
    """
    ABC for tokenizers.
    might be able to do fancy things, 
    like kicking out low frequency tokens, or encode repetitive patterns such as !!! -> ! x3 or so.
    """

    def __init__(self, delimiter: str):
        self.delimiter = delimiter

    @abc.abstractmethod
    def tokenize(self, text: str) -> List[str]:
        pass

    def tokenize_all(self, texts: ListOfRawTexts) -> List[List[str]]:
        return [self.tokenize(text) for text in texts]

    @abc.abstractmethod
    def untokenize(self, tokens: ListOfRawTexts) -> str:
        """reconstruct original data as close as possible"""
        pass


class RegexTokenizer(Tokenizer):
    def __init__(self, delimiter: str, delimiter_regex: str = None):
        super().__init__(delimiter)
        self.delimiter_regex = delimiter if delimiter_regex is None else delimiter_regex

    def tokenize(self, text: str) -> List[str]:
        return add_bos(regex.split(pattern=self.delimiter_regex, string=text))

    def untokenize(self, tokens: ListOfRawTexts) -> str:
        return self.delimiter.join(tokens)


class SimpleWordTokenizer(RegexTokenizer):
    def __init__(self):
        super().__init__(delimiter=" ", delimiter_regex="\s+")


class CharTokenizer(RegexTokenizer):
    def __init__(self):
        super().__init__(delimiter="")


class FastAITokenizer(Tokenizer):

    def __init__(self, lang: str, delimiter: str = " "):
        super().__init__(delimiter)
        self._tokenizer = transform.Tokenizer(lang=lang, post_rules=transform.defaults.text_post_rules + [add_bos])

    def tokenize(self, text: str) -> List[str]:
        return self._tokenizer.process_all(texts=[text])[0]

    def untokenize(self, tokens: ListOfRawTexts) -> str:
        return self.delimiter.join(tokens)

    def tokenize_all(self, texts: ListOfRawTexts, max_len: int = None) -> List[List[str]]:
        return self._tokenizer.process_all(texts=texts)


class SequenceMapper:
    def __init__(self, vocab: transform.Vocab):
        self.vocab = vocab
        self.vocab_size = len(vocab.stoi)

    def encode_texts(self, tokenized_texts: ListOfTokenizedTexts,
                     length: int = None) -> List[List[int]]:
        def encode_single_text(tokens):
            if not length:
                return self.vocab.numericalize(tokens)
            else:
                indices = np.full(length, fill_value=self.vocab.stoi[transform.PAD])
                for i, token_index in enumerate(self.vocab.numericalize(tokens)[:length]):
                    indices[i] = token_index
                return indices

        encoded = [encode_single_text(t) for t in tokenized_texts]
        return encoded

    def decode_texts(self, indices: List[List[int]], delimiter: str = " ") -> ListOfRawTexts:
        return [self.vocab.textify(nums=single_sequence_indices, sep=delimiter) for single_sequence_indices in indices]

    def encode_texts_onehots(self,
                             tokenized_texts: ListOfTokenizedTexts,
                             length: int) -> Tensor:
        indices = np.array(self.encode_texts(tokenized_texts=tokenized_texts,
                                             length=length))

        onehots = np.zeros((indices.shape[0], indices.shape[1], self.vocab_size))
        onehots[np.arange(onehots.shape[0]), indices] = 1
        return onehots


def generate_stateful_lm_data(sm: SequenceMapper, tokenized_texts: transform.Collection[transform.Collection[str]],
                              minibatch_size: int, backprop_length: int) -> np.ndarray:
    """
    generate training data of concatenated texts so that the first item of batch n is the continuation of the first item of batch n-1 etc.
    :param sm: sequence mapper to use
    :param tokenized_texts: tokenized texts to use
    :param minibatch_size:
    :param backprop_length: backpropagation through time
    :return: np.ndarray of shape [n_batches,minibatch_size, backprop_length] containing token indices
    """

    text_length = backprop_length + 1

    encoded_texts = sm.encode_texts(tokenized_texts=tokenized_texts)

    flattened_tokens = [token for text in encoded_texts for token in text]
    n_tokens = len(flattened_tokens)

    # initialize index array with padding value
    n_minibatches = n_tokens // (minibatch_size * text_length)
    indices = np.full(shape=(n_minibatches, minibatch_size, text_length), fill_value=sm.vocab.stoi[transform.PAD])

    # fill it! yay

    chunks = chunking(flattened_tokens, chunk_size=text_length)

    for i_minibatch_entry in range(minibatch_size):
        for i_minibatch in range(n_minibatches):
            entry_tokens = next(chunks)
            # print(f"n_tokens {n_tokens} n_minibatches {n_minibatches} i_minibatch {i_minibatch} i_minibatch_entry {i_minibatch_entry} tokens {sm.decode_texts([entry_tokens])}")
            indices[i_minibatch, i_minibatch_entry] = entry_tokens
    return indices


if __name__ == "__main__":
    corpus = Corpus.fromJSON(
        path="/Users/marlon/Code/Python/pm-linguistic-deep-learning/data/intent_data/old_en.json",
        normalize_x=True) \
        .filter_for_labels(
        labels={-1, 6379, 443, 274, 266, 22, 261, 414, 720, 430, 337, 335, 6374, 415, 57, 269, 283, 369, 2, 18, 292,
                425,
                402, 6367, 264, 440, 408, 265, 436, 704, 344, 360, 397, 36, 347},
        junk_label=-1)

    tok = FastAITokenizer(lang="en")
    tokenized_texts = tok.tokenize_all(corpus.X)

    vocab = transform.Vocab.create(tokens=tokenized_texts, max_vocab=1000, min_freq=2)
    sm = SequenceMapper(vocab=vocab)
    indices = generate_stateful_lm_data(sm=sm, tokenized_texts=tokenized_texts, minibatch_size=3, backprop_length=10)
