import abc
from typing import Dict, List

import numpy as np
import torch as t
from torch.distributions import Categorical
from torch.nn import CrossEntropyLoss

from unify_eval.utils.vocab import PAD
from unify_eval.model.mixins.stateful import StatefulModel
from unify_eval.model.types import Tensor
from unify_eval.training.seq2seq.seq2seq_data import Seq2SeqData
from unify_eval.utils.text_sequence import SequenceMapper, Tokenizer


class Sequence2SequenceModel(StatefulModel):
    """
    class for modeling stateful sequence 2 sequence models
    """

    def __init__(self, sequence_mapper: SequenceMapper, tokenizer: Tokenizer):
        StatefulModel.__init__(self)
        self.sequence_mapper = sequence_mapper
        self.tokenizer = tokenizer
        self.xent = CrossEntropyLoss(ignore_index=self.sequence_mapper.vocab.token2id[PAD])

    @abc.abstractmethod
    def train(self, data: Seq2SeqData, **kwargs) -> "Sequence2SequenceModel":
        pass

    @abc.abstractmethod
    def predict_target_logits(self, indices: List[List[int]]) -> Tensor:
        """
        returns logits for target sequence
        :param indices: indices of input text tokens
        :return: some tensor of shape [n_minibatch,seq_length,vocab_size]
        """
        pass

    @abc.abstractmethod
    def predict_target_logprobs(self, indices: List[List[int]]) -> Tensor:
        """
        returns logprobs for target sequence
        :param indices: indices of input text tokens
        :return: some tensor of shape [n_minibatch,seq_length,vocab_size]
        """
        pass

    @abc.abstractmethod
    def predict_target_probs(self, indices: List[List[int]]) -> Tensor:
        """
        returns probs for target sequence
        :param indices: indices of input text tokens
        :return: some tensor of shape [n_minibatch,seq_length,vocab_size]
        """
        pass

    @abc.abstractmethod
    def _get_cross_entropy_singlebatch(self, input_indices: List[List[int]], target_indices: List[List[int]],
                                       batch_first: bool = False, **kwargs) -> Tensor:
        """
        calculates cross entropy of a given data set (base e!)
        :param input_indices:
        :param target_indices:
        :param kwargs:
        :return:
        """
        pass

    def get_cross_entropy(self, data: Seq2SeqData, batch_first: bool = False, reduction: str = "mean",
                          **kwargs) -> Tensor:
        """
        calculates cross entropy of a given data set (base e!)
        :param input_indices:
        :param target_indices:
        :param reduction: either "mean" or "sum"
        :param kwargs:
        :return:
        """
        xents = []
        for input_data, target_data in data:
            xents.append(self._get_cross_entropy_singlebatch(input_indices=input_data,
                                                             target_indices=target_data,
                                                             batch_first=batch_first,
                                                             **kwargs).detach().item())
        return np.sum(xents) if reduction == "sum" else np.mean(xents)

    def get_perplexity(self, data: Seq2SeqData, batch_first: bool = False, reduction: str = "mean", **kwargs) -> Tensor:
        """
        calculates perplexity of a given data set (base e!)
        :param input_indices:
        :param target_indices:
        :param reduction: either "mean" or "sum"
        :param kwargs:
        :return:
        """

        return np.exp(self.get_cross_entropy(data=data, batch_first=batch_first, reduction=reduction, **kwargs))


class PytorchSequence2SequenceModel(Sequence2SequenceModel):
    """
    class for modeling sequence 2 sequence models
    """

    @abc.abstractmethod
    def predict_target_logits(self, indices: List[List[int]]) -> t.Tensor:
        """
        returns logits for target sequence
        :param indices: indices of input text tokens
        :return: some tensor of shape [n_minibatch,seq_length,vocab_size]
        """
        pass

    def predict_target_logprobs(self, indices: List[List[int]]) -> t.Tensor:
        """
        returns logprobs for target sequence
        :param indices: indices of input text tokens
        :return: some tensor of shape [n_minibatch,seq_length,vocab_size]
        """
        logits = self.predict_target_logits(indices=indices)
        return t.nn.LogSoftmax(dim=-1)(logits)

    def predict_target_probs(self, indices: List[List[int]]) -> t.Tensor:
        """
        returns probs for target sequence
        :param indices: indices of input text tokens
        :return: some tensor of shape [n_minibatch,seq_length,vocab_size]
        """
        logits = self.predict_target_logits(indices=indices)
        return t.nn.Softmax(dim=-1)(logits)

    def _get_cross_entropy_singlebatch(self, input_indices: List[List[int]], target_indices: List[List[int]],
                                       batch_first: bool = False, **kwargs) -> t.Tensor:
        """
        calculates cross entropy of a given data set (base e!)
        :param input_indices:
        :param target_indices:
        :param kwargs:
        :return:
        """
        logits = self.predict_target_logits(indices=input_indices)
        padded_indices = t.nn.utils.rnn.pad_sequence(
            sequences=[t.tensor(indices_).long() for indices_ in target_indices],
            padding_value=self.sequence_mapper.vocab.stoi["xxpad"],
            batch_first=batch_first)

        cross_entropy = self.xent(logits.view(-1, logits.shape[-1]), target=padded_indices.view(-1, ))
        return cross_entropy


class LanguageModel(Sequence2SequenceModel):
    """
    class for Language models. Expects LMs to be Sequence2Sequence models, with the target sequence being the input sequence shifted by one token
    """

    @abc.abstractmethod
    def get_loss(self, input_indices: List[List[int]], target_indices: List[List[int]], **kwargs) -> Dict[str, Tensor]:
        pass

    def generate(self, text: str, n_tokens: int, temperature: float = 1.0, **kwargs) -> str:
        token_indices = self.sequence_mapper.encode_texts([self.tokenizer.tokenize(text=text)])[0]

        generated_indices = self._generate_indices(token_indices=token_indices,
                                                   n_tokens=n_tokens,
                                                   temperature=temperature,
                                                   **kwargs)

        return self.sequence_mapper.decode_texts(encoded_texts=[token_indices + generated_indices],
                                                 delimiter=self.tokenizer.delimiter)[0]

    @abc.abstractmethod
    def _generate_indices(self, token_indices: List[int], n_tokens: int, current_state: Tensor,
                          temperature: float = 1.0, **kwargs) -> List[int]:
        pass


class PytorchLanguageModel(LanguageModel, PytorchSequence2SequenceModel):
    """
    class for Language models. Expects LMs to be Sequence2Sequence models, with the target sequence being the input sequence shifted by one token
    """

    def _generate_indices(self, token_indices: List[int], n_tokens: int,
                          temperature: float = 1.0, **kwargs) -> List[int]:
        """
        generate a sequence of a given length
        :param token_indices: start of text
        :param n_tokens: length of generated text
        :param temperature: softmax temperature
        :param kwargs: additional kwargs passed to model
        :return: generated text
        """

        if n_tokens <= 0:
            return token_indices
        logits = self.predict_target_logits(indices=[token_indices])[0, -1] / temperature

        sampled_index = Categorical(logits=logits).sample().detach().item()

        return [sampled_index] + self._generate_indices(token_indices=[sampled_index], n_tokens=n_tokens - 1,
                                                        temperature=temperature, **kwargs)
