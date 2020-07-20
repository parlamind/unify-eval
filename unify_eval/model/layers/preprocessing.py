from typing import List
from unify_eval.model.layers.layer_base import Layer
from unify_eval.utils.text_sequence import SequenceMapper, Tokenizer
from unify_eval.utils.vocab import Vocab, PAD


class TokenizerLayer(Tokenizer, Layer):
    """
    Wrapper around a particular Tokenizer instance.
    """

    def __init__(self, tokenizer: Tokenizer):
        Tokenizer.__init__(self, tokenizer.delimiter)
        Layer.__init__(self)
        self.tokenizer = tokenizer

    def tokenize(self, text: str) -> List[str]:
        return self.tokenizer.tokenize(text=text)

    def tokenize_all(self, texts: List[str], max_len: int = None) -> List[List[str]]:
        return self.tokenizer.tokenize_all(texts=texts)

    def untokenize(self, tokens: List[str]) -> str:
        return self.tokenizer.untokenize(tokens=tokens)

    def push(self, **kwargs) -> dict:
        text_kw = kwargs["text_kw"]
        tokenized = self.tokenizer.tokenize_all(texts=kwargs[text_kw])
        kwargs["tokenized_texts"] = tokenized
        return kwargs

    def get_components(self) -> dict:
        return dict(tokenizer=self.tokenizer)

    @classmethod
    def from_components(cls, **kwargs) -> "TokenizerLayer":
        return TokenizerLayer(**kwargs)


class SequenceMapperLayer(SequenceMapper, Layer):
    """
    Wrapper around a SequenceMapper instance.
    """

    def __init__(self, vocab: Vocab):
        SequenceMapper.__init__(self, vocab)
        Layer.__init__(self)

    def push(self, **kwargs) -> dict:
        kwargs["encoded_texts"] = self.encode_texts(tokenized_texts=kwargs["tokenized_texts"])
        kwargs["padding_value"] = self.vocab.token2id[PAD]
        return kwargs

    def get_components(self) -> dict:
        return {"vocab": self.vocab}

    @classmethod
    def from_components(cls, **kwargs) -> "Layer":
        return SequenceMapperLayer(vocab=kwargs["vocab"])
