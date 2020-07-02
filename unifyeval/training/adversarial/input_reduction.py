from dataclasses import dataclass
from typing import List

import numpy as np

from unifyeval.model.mixins.adversarial import SaliencyModel
from unifyeval.model.types import Label
from unifyeval.training.callback import TrainerCallback
from unifyeval.utils.load_data import KeyedBatchLoader
from unifyeval.utils.text_sequence import Tokenizer


def reduce_input(tokens: List[str],
                 saliency_vector: np.ndarray,
                 tokenizer: Tokenizer) -> str:
    if len(tokens) == 0:
        return ""
    i_lowest = saliency_vector[:len(tokens)].argmin(axis=-1)
    return tokenizer.untokenize(tokens[:i_lowest] + tokens[i_lowest + 1:] if i_lowest - 1 < len(tokens) else [])


@dataclass
class InputReduction:
    """
    data class storing input reduction results
    """
    tokens: List[str]
    saliency: List[float]
    original_label: Label
    original_label_prob: float
    current_label: Label
    current_label_prob: float


def get_all_reductions(text: str,
                       text_kw: str,
                       model: SaliencyModel,
                       label: int,
                       tokenizer: Tokenizer,
                       **kwargs) -> List[InputReduction]:
    tokens = tokenizer.tokenize(text)
    if len(tokens) == 0:
        return []
    saliency = model.get_saliency(texts=[text],
                                  label=label,
                                  max_length=kwargs["max_length"])[0]

    reduced = reduce_input(tokens, saliency_vector=np.array(saliency), tokenizer=tokenizer)
    label_probs = model.predict_label_probabilities(**{text_kw: [text]}, **kwargs)[0]
    new_label_index = np.argmax(label_probs)
    old_label_prob = label_probs[model.label_mapper.actuallabel2index[label]]
    new_label_prob = label_probs[new_label_index]
    return [InputReduction(tokens=tokens,
                           saliency=saliency,
                           original_label=label,
                           original_label_prob=old_label_prob,
                           current_label=model.label_mapper.index2actuallabel[new_label_index],
                           current_label_prob=new_label_prob)] + get_all_reductions(
        text=reduced,
        text_kw=text_kw,
        model=model,
        label=label,
        tokenizer=tokenizer,
        **kwargs)


class InputReductionCallback(TrainerCallback):
    """
    WIP callback to print input reductions. basis for actual visualization
    """

    def __init__(self, data_loader: KeyedBatchLoader,
                 text_kw: str = "text",
                 label_kw: str = "label"):
        """
        :param data_loader: data loader, duh
        :param text_kw: keyword for text in batch
        :param label_kw: keyword for label in batch
        """
        self.data_loader = data_loader
        self.text_kw = text_kw
        self.label_kw = label_kw

    def __call__(self, model: SaliencyModel, i_minibatch: int, iteration: int, *args, **kwargs):

        all_reductions = []
        for batch in self.data_loader.yield_minibatches(minibatch_size=16, progress_bar=True):
            for text, label in zip(batch[self.text_kw], batch[self.label_kw]):
                all_reductions.append(get_all_reductions(text=text,
                                                         text_kw=self.text_kw,
                                                         model=model,
                                                         label=label,
                                                         tokenizer=model.sequence_mapper.tokenizer,
                                                         **kwargs))
        for reductions in all_reductions:
            print("-----------------------------------")
            for single_reduction in reductions:
                print(f"original label {single_reduction.original_label} {single_reduction.original_label_prob}")
                print(f"current label {single_reduction.current_label} {single_reduction.current_label_prob}")
                for token, saliency in zip(single_reduction.tokens, single_reduction.saliency):
                    print(token, saliency)
                print("-----------")
