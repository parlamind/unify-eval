import abc
from abc import ABC
from typing import Dict, List

import torch as t

from unify_eval.model.deep_model import DeepModel
from unify_eval.model.mixins.sequences.seq_input import SequenceInputModel
from unify_eval.model.types import Tensor, ListOfRawTexts
from unify_eval.utils.text_sequence import SequenceMapper, Tokenizer


class WhiteboxModel(DeepModel):
    """
    Class for models that allow for gradient calculation
    """

    @abc.abstractmethod
    def get_gradients(self, tensor: Tensor, with_respect_to: Tensor, **kwargs) -> Tensor:
        """
        get gradients of a tensor with respect to another one
        :param tensor: some tensor (domain of partial derivative)
        :param with_respect_to: another tensor (codomain of partial derivative)
        :param kwargs: additional kwargs
        :return: Tensor instance of same shape as "tensor"
        """
        pass


class PyTorchWhiteboxModel(WhiteboxModel):
    """
    Subclass of WhiteboxModel with get_gradients implemented for pytorch models
    """

    @abc.abstractmethod
    def get_module(self) -> t.nn.Module:
        """
        returns torch.nn.Module instance containing all parameters used for a given model
        (e.g. for zeroing gradients back after single gradient calculations)
        :return: torch.nn.Module instance
        """
        pass

    def get_gradients(self, tensor: t.Tensor, with_respect_to: t.Tensor, **kwargs) -> t.Tensor:
        """
        get gradients of a tensor with respect to another one.
        :param tensor: some tensor (domain of partial derivative)
        :param with_respect_to: another tensor (codomain of partial derivative)
        :param kwargs: additional kwargs,
        including necessary "module" containing t.nn.Module instance with tensor and with_respect_to as nodes
        :return: Tensor instance of same shape as "tensor"
        """

        module: t.nn.Module = kwargs["module"]
        with_respect_to.retain_grad()
        tensor.backward()
        grad = with_respect_to.grad.detach().clone()
        # set all gradients back to zero
        module.zero_grad()
        return grad


class SaliencyModel(SequenceInputModel, WhiteboxModel):
    """
    Class for sequential classifier models that can yield a saliency over the input
    """

    def __init__(self, tokenizer: Tokenizer, sequence_mapper: SequenceMapper):
        SequenceInputModel.__init__(self,
                                    tokenizer=tokenizer,
                                    sequence_mapper=sequence_mapper)

    @abc.abstractmethod
    def get_saliency_matrix(self, texts: ListOfRawTexts, label: int, max_length: int = None, **kwargs) -> Tensor:
        """
        returns saliency for a given input
        :param texts: texts to check
        :param label: label to query saliency for
        :param max_length: maximum length of input. Used by tokenizer
        :param kwargs: additional kwargs
        :return: matrix of shape [minibatch_size,max_length]
        """
        pass

    def get_saliency(self, texts: ListOfRawTexts, label: int, max_length: int = None, **kwargs) -> List[List[float]]:
        """
        returns token-wise saliency given some label index
        :param texts: texts to check
        :param label: label to query saliency for
        :param max_length: maximum length of input. Used by tokenizer
        :param kwargs: additional kwargs
        :return: list if lists of floats, one for every token
        """
        collected = []
        for tokenized_text, saliency_row in zip(self.tokenizer.tokenize_all(texts=texts),
                                                self.get_saliency_matrix(texts=texts,
                                                                         label=label,
                                                                         max_length=max_length,
                                                                         **kwargs)):
            tokenized_text = tokenized_text[:max_length]
            saliency_tmp = []
            for token, saliency in zip(tokenized_text, saliency_row):
                saliency_tmp.append(saliency)
            collected.append(saliency_tmp)
        return collected

    def print_saliency(self, texts: ListOfRawTexts, label: int, max_length: int = None, **kwargs) -> "SaliencyModel":
        """
        prints token-wise saliency, given some label
        :param texts: texts to check
        :param label: label to query saliency for
        :param max_length: maximum length of input. Used by tokenizer
        :param kwargs: additional kwargs
        :return: current SaliencyModel instance
        """
        print("-------------------")
        for text, tokenized_text, saliency_row in zip(texts,
                                                      self.tokenizer.tokenize_all(texts=texts),
                                                      self.get_saliency_matrix(texts=texts,
                                                                               label=label,
                                                                               max_length=max_length,
                                                                               **kwargs)):
            tokenized_text = tokenized_text[:max_length]
            print(text)
            for token, saliency in zip(tokenized_text, saliency_row):
                print(token, saliency)
            print("-------------------")
        return self

    @abc.abstractmethod
    def get_loss_from_embeddings(self, embeddings: Tensor, **kwargs) -> Dict[str, Tensor]:
        pass

    @abc.abstractmethod
    def get_loss_from_onehots(self, onehots: Tensor, **kwargs) -> Dict[str, Tensor]:
        pass


class PytorchSaliencyModel(SaliencyModel, PyTorchWhiteboxModel, ABC):
    def __init__(self, tokenizer: Tokenizer, sequence_mapper: SequenceMapper):
        SaliencyModel.__init__(self,
                               tokenizer=tokenizer,
                               sequence_mapper=sequence_mapper)
        PyTorchWhiteboxModel.__init__(self)
