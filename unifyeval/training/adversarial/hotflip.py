from dataclasses import dataclass
from enum import Enum
from typing import List, Set, Union, Optional

import numpy as np
import torch as t

from unifyeval.model.deep_model import DeepModel
from unifyeval.model.mixins.adversarial import PyTorchWhiteboxModel, SaliencyModel, PytorchSaliencyModel
from unifyeval.model.types import Label
from unifyeval.training.callback import TrainerCallback
from unifyeval.training.trainer import Trainer
from unifyeval.utils.annotations import auto_repr
from unifyeval.utils.load_data import KeyedBatchLoader


class Action(Enum):
    """
    enum for changes to input in order to create adversarial examples
    """
    ADD = "ADD"
    DELETE = "DELETE"
    FLIP = "FLIP"


def get_onehots(model: Union[PyTorchWhiteboxModel, SaliencyModel],
                text: str, **kwargs) -> t.Tensor:
    """
    generates onehot encoding of textual input
    :param model: model to use
    :param text: input as text
    :param kwargs: additional kwargs for model
    :return: onehot-encoded text as pytorch tensor of shape [1,max_length,vocab_size]
    """
    max_length = kwargs["max_length"]
    onehots = t.from_numpy(model.map_to_onehot_sequence(tokenized_texts=[text],
                                                        max_length=max_length)).float().requires_grad_(True)
    return onehots


def get_indices(model: Union[PyTorchWhiteboxModel, SaliencyModel],
                text: str, **kwargs) -> t.Tensor:
    """
    generates tensor containing token indices from text
    :param model: model to use
    :param text: textual input
    :param kwargs: additional kwargs for model
    :return: pytorch tensor containing input mapped to respective token indices of shape [1,max_length]
    """
    max_length = kwargs["max_length"]
    indices = t.from_numpy(model.map_to_index_sequence(tokenized_texts=[text], max_length=max_length)).long()
    return indices


@dataclass
class HotflipData:
    """
    data class that stores utilities for generating adversarial examples
    """
    # tokenizes string
    tokens: List[str]
    # token indices
    indices: np.ndarray
    # token onehots
    onehots: np.ndarray
    # loss of respective input
    loss: float
    # flattened gradients with shape [n_tokens * vocab_size]
    gradients_flat: np.ndarray

    @staticmethod
    def make(model: PytorchSaliencyModel,
             text: str,
             label: Label,
             loss_name: str = "cross_entropy",
             **kwargs) -> "HotflipData":
        """
        generate hotflip data from a given model, text and label
        :param model: model to use
        :param text: textual input
        :param label: label to use
        :param kwargs: additional kwargs
        :return: HotflipData
        """
        max_length = kwargs["max_length"]
        tokens = model.tokenizer.tokenize(text=text)
        indices = t.from_numpy(model.map_to_index_sequence(tokenized_texts=[tokens], max_length=max_length)).long()

        # onehot input needed for their respective gradients
        onehots = t.from_numpy(model.map_to_onehot_sequence(tokenized_texts=[tokens],
                                                            max_length=max_length)).float().requires_grad_(True)

        # calculate loss via onehots and indices (indices needed for indicating length of input)
        loss = model.get_loss_from_onehots(onehots=onehots, indices=indices, labels=[label])[loss_name]

        # calculate gradients and turn them into numpy array
        gradients_flat = model.get_gradients(loss, with_respect_to=onehots,
                                             module=model.get_module()).view((1, -1)).detach().numpy()

        # pack everything into data class
        return HotflipData(tokens=tokens,
                           indices=indices.detach().numpy(),
                           onehots=onehots.detach().numpy(),
                           loss=float(loss),
                           gradients_flat=gradients_flat)


def delete(model: PytorchSaliencyModel,
           text: str,
           label: Label, **kwargs) -> Optional[str]:
    """
    Finds input that maximizes loss for a given label by removing a single character.
    :param model: model to use
    :param text: input as text
    :param label: label to maximize loss for
    :param kwargs: additional kwargs for model
    :return: string if something was found, otherwise None
    """

    def reduce_input(tokens: List[str], index: int) -> str:
        """
        removes a token from a given list of tokens and concatenates the rest
        :param tokens: list of tokens
        :param index: index where token should be removed
        :return: concatenated remaining tokens
        """
        updated_tokens = tokens[:index] + tokens[index + 1:] if index - 1 < len(tokens) else []
        return model.tokenizer.untokenize(updated_tokens)

    max_length = kwargs["max_length"]
    # get gradients of loss with respect to onehot input
    hotflip_data = HotflipData.make(model=model, text=text, label=label, **kwargs)

    # loop over all positional indices to remove the token for and update surrogate loss in case
    best_position_index = 0
    initial_surrogate_loss = 0.0
    max_loss = initial_surrogate_loss

    found_something = False

    for start_index in range(len(hotflip_data.tokens) - 1):
        # create fake onehot input by modifying true input
        swap_mask = hotflip_data.onehots.copy()
        for actual_position in range(start_index, max_length - 1):
            # flip sign of true token
            swap_mask[0, actual_position, hotflip_data.indices[0, actual_position]] = -1
            # set sign of candidate position to 1
            swap_mask[0, actual_position, hotflip_data.indices[0, actual_position + 1]] = 1

        # flatten fake input
        swap_mask_flat = swap_mask.reshape((-1, max_length * model.sequence_mapper.vocab_size))
        # normalize
        swap_mask_flat = swap_mask_flat / np.linalg.norm(swap_mask_flat)
        # take dot product as single numpy scalar
        surrogate_loss = (hotflip_data.gradients_flat @ swap_mask_flat.T).item()
        # if surrogate loss is higher than before, it becomes the new loss
        if surrogate_loss > max_loss:
            max_loss = surrogate_loss
            best_position_index = start_index
            found_something = True

    if found_something:
        return reduce_input(tokens=hotflip_data.tokens, index=best_position_index)
    else:
        return None


def flip(model: PytorchSaliencyModel,
         text: str,
         label: Label,
         **kwargs) -> Optional[str]:
    """
    Finds input that maximizes loss for a given label by flipping a single character.
    Quite costly, O(input_length * vocab_size)
    :param model: model to use
    :param text: input as text
    :param label: label to maximize loss for
    :param kwargs: additional kwargs for model
    :return: string if something was found, otherwise None
    """
    max_length = kwargs["max_length"]

    # get gradients of loss with respect to onehot input
    hotflip_data = HotflipData.make(model=model, text=text, label=label, **kwargs)

    # loop over all positions and possible tokens in vocabulary to find optimal input
    best_position_index = 0
    best_vocab_index = 0
    initial_surrogate_loss = 0.0
    max_loss = initial_surrogate_loss

    found_something = False

    for i_position in range(len(hotflip_data.tokens)):
        for i_vocab in range(model.sequence_mapper.vocab_size):
            # create fake onehot input by modifying true input
            swap_mask = hotflip_data.onehots.copy()
            # flip sign of true token
            swap_mask[0, i_position, hotflip_data.indices[0, i_position]] = -1
            # set sign of candidate position / vocab item to 1
            swap_mask[0, i_position, i_vocab] = 1
            # flatten input
            swap_mask_flat = swap_mask.reshape((-1, max_length * model.sequence_mapper.vocab_size))
            # take dot product as single numpy scalar
            surrogate_loss = (hotflip_data.gradients_flat @ swap_mask_flat.T).item()
            # if surrogate loss is higher than before, it becomes the new loss
            if surrogate_loss > max_loss:
                found_something = True
                max_loss = surrogate_loss
                best_position_index = i_position
                best_vocab_index = i_vocab
    # if anything was found, update tokens, create input string and return it
    if found_something:
        updated_tokens = list(hotflip_data.tokens)
        updated_tokens[best_position_index] = model.sequence_mapper.vocab.itos[best_vocab_index]
        return model.tokenizer.untokenize(updated_tokens)
    else:
        return None


def add(model: PytorchSaliencyModel,
        text: str,
        label: Label,
        **kwargs) -> Optional[str]:
    """
    finds input that maximizes loss for a given label by adding a single character.
    Very costly.
    :param model: model to use
    :param text: input as text
    :param label: label to maximize loss for
    :param kwargs: additional kwargs for model
    :return: string if something was found, otherwise None
    """

    # take maximum length of input
    max_length = kwargs["max_length"]

    # get gradients of loss with respect to onehot input
    hotflip_data = HotflipData.make(model=model, text=text, label=label, **kwargs)

    # loop over all positions and possible tokens in vocabulary to find optimal input
    best_position_index = 0
    best_vocab_index = 0
    initial_surrogate_loss = 0.0
    max_loss = initial_surrogate_loss

    found_something = False

    for insertion_index in range(len(hotflip_data.tokens)):

        for vocab_index in range(model.sequence_mapper.vocab_size):

            # create fake onehot input by modifying true input
            swap_mask = hotflip_data.onehots.copy()

            # flip sign of true token
            swap_mask[0, insertion_index, hotflip_data.indices[0, insertion_index]] = -1

            # set sign of candidate position / vocab item to 1
            swap_mask[0, insertion_index, vocab_index] = 1

            # if you have additional tokens following, shift all to the right
            if insertion_index < len(hotflip_data.tokens) - 1:

                for shift_position in range(insertion_index + 1, min(len(hotflip_data.tokens) + 1, max_length)):
                    swap_mask[0, shift_position, hotflip_data.indices[0, shift_position]] = -1
                    swap_mask[0, shift_position, hotflip_data.indices[0, shift_position - 1]] = 1

            # flatten input
            swap_mask_flat = swap_mask.reshape((-1, max_length * model.sequence_mapper.vocab_size))
            # normalize by respective l2 norm
            swap_mask_flat = swap_mask_flat / np.linalg.norm(swap_mask_flat)
            # take dot product as single numpy scalar
            surrogate_loss = (hotflip_data.gradients_flat @ swap_mask_flat.T).item()
            # print(i_vocab,surrogate_loss)

            # if surrogate loss is higher than before, it becomes the new loss
            if surrogate_loss > max_loss:
                found_something = True
                max_loss = surrogate_loss
                best_position_index = insertion_index
                best_vocab_index = vocab_index

    # if anything was found, update tokens, create input string and return it
    if found_something:
        updated_tokens = hotflip_data.tokens[:best_position_index + 1] \
                         + [model.sequence_mapper.vocab.itos[best_vocab_index]] \
                         + hotflip_data.tokens[best_position_index + 1:]
        return model.tokenizer.untokenize(updated_tokens)
    else:
        return None


def maybe_generate_adversarial_example(model: PytorchSaliencyModel,
                                       text: str,
                                       label: Label,
                                       action: Action,
                                       **kwargs) -> Optional[str]:
    """
    for a given action (delete, add, flip single characters), try to change input so that it maximizes the loss
    :param model: model to use
    :param text: input as raw text
    :param label: true label
    :param action: Action.ADD, Action.DELETE or Action.FLIP
    :param kwargs: additional kwargs
    :return: string if some input is found, otherwise None
    """
    return {
        Action.ADD: add,
        Action.DELETE: delete,
        Action.FLIP: flip
    }[action](model=model, text=text, label=label, **kwargs)


@auto_repr
class HotflipTrainer(Trainer):
    """
    Trainer for hotflip training. Given a set of changes to the input (character-wise flip, deletion and addition),
    it finds the one that maximizes loss and trains against it
    """

    def __init__(self, data_loader: KeyedBatchLoader,
                 minibatch_callbacks: List[TrainerCallback],
                 batch_callbacks: List[TrainerCallback],
                 hotflip_actions: Set[Action],
                 adversarial_data_kw: str,
                 label_kw: str):
        """

        :param data_loader: training data
        :param minibatch_callbacks:  callbacks to run after every minibatch
        :param batch_callbacks: callbacks to run after every full batch
        :param hotflip_actions: set of actions used to create adversarial examples
        :param adversarial_data_kw:
        :param label_kw:
        """

        Trainer.__init__(self, data_loader, minibatch_callbacks, batch_callbacks)
        self.hotflip_actions = hotflip_actions
        self.adversarial_data_kw = adversarial_data_kw
        self.label_kw = label_kw

    def train_on_minibatch(self, model: PytorchSaliencyModel, keyed_minibatch: dict, i_minibatch: int, iteration: int,
                           **kwargs) -> DeepModel:

        # train on data as usual
        super().train_on_minibatch(model, keyed_minibatch, i_minibatch, iteration, **kwargs)
        # batch to store adversarial data
        adversarial_batch = dict((key, []) for key in keyed_minibatch)
        other_keys = {key for key in adversarial_batch if key not in {self.adversarial_data_kw, self.label_kw}}

        single_keyed_datapoints = list(dict((key, val)
                                            for key, val in zip(keyed_minibatch.keys(), vals))
                                       for vals in zip(*keyed_minibatch.values()))

        # for every datapoint, check if an adversarial example can be found
        for datapoint in single_keyed_datapoints:
            text = datapoint[self.adversarial_data_kw]
            label = datapoint[self.label_kw]

            for action in self.hotflip_actions:

                maybe_adversarial = maybe_generate_adversarial_example(model=model,
                                                                       text=text,
                                                                       label=label,
                                                                       action=action, **kwargs)
                # if something was found, add it to adversarial batch
                if maybe_adversarial:

                    for key, value in datapoint.items():
                        if key in other_keys:
                            adversarial_batch[key].append(value)
                    adversarial_batch[self.label_kw].append(label)
                    adversarial_batch[self.adversarial_data_kw].append(maybe_adversarial)
        # turn values from lists into numpy arrays
        adversarial_batch = dict((key, np.array(values)) for key, values in adversarial_batch.items())
        # make sure you always take at least 2 data points, otherwise batch norm will break due to standard deviation being 0
        if len(adversarial_batch[self.label_kw]) > 2:
            super().train_on_minibatch(model,
                                       adversarial_batch,
                                       i_minibatch,
                                       iteration,
                                       **kwargs)

        return model
