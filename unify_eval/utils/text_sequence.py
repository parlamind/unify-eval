import abc
from abc import ABC
from typing import List

import numpy as np
from regex import regex

from unify_eval.utils.iter_utils import chunking
from unify_eval.model.types import Tensor, ListOfRawTexts, ListOfTokenizedTexts
from unify_eval.utils.vocab import Vocab, PAD, BOS


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
        return regex.split(pattern=self.delimiter_regex, string=text)

    def untokenize(self, tokens: ListOfRawTexts) -> str:
        return self.delimiter.join(tokens)


class SimpleWordTokenizer(RegexTokenizer):
    def __init__(self):
        super().__init__(delimiter=" ", delimiter_regex=r"\s+")


class CharTokenizer(RegexTokenizer):
    def __init__(self):
        super().__init__(delimiter="")


class SequenceMapper:
    def __init__(self, vocab: Vocab):
        self.vocab = vocab
        self.vocab_size = len(vocab)

    def encode_texts(self, tokenized_texts: ListOfTokenizedTexts,
                     length: int = None) -> List[List[int]]:
        def encode_single_text(tokens: List[str]):
            # add bos
            tokens = [BOS] + tokens
            token_ids = [self.vocab.encode_token(token) for token in tokens]

            if not length:
                return token_ids
            else:
                indices = np.full(length, fill_value=self.vocab.token2id[PAD])
                for i, token_index in enumerate(token_ids[:length]):
                    indices[i] = token_index
                return indices

        encoded = [encode_single_text(t) for t in tokenized_texts]
        return encoded

    def decode_texts(self, encoded_texts: List[List[int]], delimiter: str = " ") -> ListOfRawTexts:
        return [delimiter.join([self.vocab.decode_token_index(i) for i in encoded_text]) for encoded_text in
                encoded_texts]

    def encode_texts_onehots(self,
                             tokenized_texts: ListOfTokenizedTexts,
                             length: int) -> Tensor:
        indices = np.array(self.encode_texts(tokenized_texts=tokenized_texts,
                                             length=length))

        onehots = np.zeros((indices.shape[0], indices.shape[1], self.vocab_size))
        onehots[np.arange(onehots.shape[0]), indices] = 1
        return onehots
