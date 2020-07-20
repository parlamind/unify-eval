from collections import defaultdict
from typing import List, Tuple

from unify_eval.model.types import ListOfTokenizedTexts

PAD = "xxpad"
UNK = "xxunk"
BOS = "xxbos"


class Vocab:
    def __init__(self, id2token: List[str]):
        self.id2token = id2token
        self.token2id = dict((token, i) for i, token in enumerate(self.id2token))

    @classmethod
    def make(cls,
             texts: ListOfTokenizedTexts,
             max_vocab_size: int = 10000,
             min_count: int = 1,
             special_tokens: List[Tuple[str, int]] = None,
             ) -> "Vocab":

        # add default special tokens if given list is None
        special_tokens = special_tokens \
            if special_tokens is not None \
            else [(token, i) for i, token in enumerate((PAD, UNK, BOS))]

        # collect token counts
        token2count = defaultdict(lambda: 0)
        for text in texts:
            for token in text:
                token2count[token] += 1

        # map to ids after checking agains min count
        id2token = []
        for token, count in token2count.items():
            if count >= min_count:
                id2token.append(token)

        # get rid of low frequency tokens
        sorted_tokens = sorted(id2token, key=lambda x: token2count[x], reverse=True)[:max_vocab_size]

        # add special tokens (and remove last token to keep vocab size intact)
        for special_token, special_token_index in special_tokens:
            sorted_tokens = sorted_tokens[:special_token_index] \
                            + [special_token] \
                            + sorted_tokens[special_token_index:-1]

        return Vocab(sorted_tokens)

    def encode_token(self, token: str) -> int:
        return self.token2id[token] if token in self.token2id else self.token2id[UNK]

    def decode_token_index(self, index: int) -> str:
        return self.id2token[index] if index < len(self) else UNK

    def __len__(self):
        return len(self.id2token)
