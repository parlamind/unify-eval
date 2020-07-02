from typing import List

from fastai.text import transform

from unifyeval.model.layers.layer_base import Layer
from unifyeval.utils.text_sequence import FastAITokenizer, SequenceMapper, Tokenizer


class TokenizerLayer(Tokenizer, Layer):
    """
    Wrapper around a particular Tokenizer instance.
    """
    def __init__(self, tokenizer: Tokenizer):
        super().__init__(tokenizer.delimiter)
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


class FastAITokenizerLayer(FastAITokenizer, Layer):
    """
    Wrapper around FastAITokenizer instance.
    """
    def __init__(self, lang: str):
        self.lang = lang
        super().__init__(lang)

    def push(self, **kwargs) -> dict:
        text_kw = kwargs["text_kw"]
        tokenized = self.tokenize_all(texts=kwargs[text_kw], max_len=kwargs["max_len"] if "max_len" in kwargs else None)
        kwargs["tokenized_texts"] = tokenized
        return kwargs

    def get_components(self) -> dict:
        return {"lang": self.lang}


class SequenceMapperLayer(SequenceMapper, Layer):
    """
    Wrapper around a SequenceMapper instance.
    """
    def __init__(self, vocab: transform.Vocab):
        super().__init__(vocab)

    def push(self, **kwargs) -> dict:
        kwargs["encoded_texts"] = self.encode_texts(tokenized_texts=kwargs["tokenized_texts"])
        kwargs["padding_value"] = self.vocab.stoi["xxpad"]
        return kwargs

    def get_components(self) -> dict:
        return {"vocab": self.vocab}

    @classmethod
    def from_components(cls, **kwargs) -> "Layer":
        return SequenceMapperLayer(vocab=kwargs["vocab"])
