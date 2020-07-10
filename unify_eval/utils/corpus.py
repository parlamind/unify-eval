import codecs
import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Union, Set, Tuple, Dict, List, Callable

import numpy as np
import pandas as pd

from unify_eval.model.types import Label
from unify_eval.utils.label_mapper import LabelMapper


def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"ä", "ae", s)
    s = re.sub(r"ö", "oe", s)
    s = re.sub(r"ü", "ue", s)
    s = re.sub(r"ß", "ss", s)
    return s


def group(X_data, Y_data, as_array: bool = True):
    grouped = dict()
    for x, y in zip(X_data, Y_data):
        if y not in grouped:
            grouped[y] = []
        grouped[y].append(x)
    return grouped if not as_array else dict((k, np.array(v)) for k, v in grouped.items())


def get_XY_json(path):
    print("reading data from", path)
    with codecs.open(path, "r", "utf-8") as f:
        data = json.load(f)

    clauses = [entry["clause"] for entry in data]
    ids = [entry["label"] for entry in data]
    return (clauses, ids)


@dataclass
class Clause:
    """
    data class for single clause
    """
    clause: str
    label: int


@dataclass
class Message:
    """
    data class for whole messages, with one intent per clause
    """
    clauses: List[str]
    labels: List[int]

    @staticmethod
    def from_clauses(clauses: List[Clause]) -> "Message":
        return Message(clauses=[c.clause for c in clauses],
                       labels=[c.label for c in clauses])


class MessageCorpus:
    """
    class storing whole messages, with one intent per clause
    """

    def __init__(self, messages: List[Message]):
        self.messages = messages
        self.label_mapper = LabelMapper(Y=[label for message in messages for label in message.labels],
                                        ignore_index=1000)

    @staticmethod
    def fromJSON(path: str, normalize_x: bool = False) -> "MessageCorpus":
        with open(path, "r") as f:
            data = json.load(f)

        messages = [Message.from_clauses([Clause(**entry) for entry in d["clauses"]]) for d in data]

        if normalize_x:
            messages = [Message(clauses=[normalize(clause) for clause in m.clauses], labels=m.labels) for m in messages]
        return MessageCorpus(messages=messages)

    def rewrite_by_label_size(self,
                              min_label_size: int = 10,
                              junk_label: Label = -1) -> "MessageCorpus":
        """
        rewrites labels to junk_label if the true label does not appear at least min_label_size times
        :param min_label_size:
        :param junk_label:
        :return:
        """
        counter = Counter(self.label_mapper.labels)

        return self.rewrite_labels(labels=set(label for label, size in counter.items() if size >= min_label_size),
                                   junk_label=junk_label)

    def rewrite_labels(self,
                       labels: Set[Union[str, int]],
                       junk_label: Label = -1) -> "MessageCorpus":
        """
        rewrites labels to junk_label if the true label does not appear in the given set of labels
        :param min_label_size:
        :param junk_label:
        :return:
        """
        messages = []
        for message in self.messages:
            updated_labels = [label if label in labels else junk_label for label in message.labels]
            messages.append(Message(clauses=message.clauses, labels=updated_labels))
        return MessageCorpus(messages=messages)

    def get_data(self, datapoint_indices: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        messages_array = np.array(self.messages)
        datapoint_indices = np.arange(messages_array.shape[0]) if datapoint_indices is None else datapoint_indices
        extracted_messages = messages_array[datapoint_indices]
        clauses = np.array([message.clauses for message in extracted_messages])
        labels = np.array([message.labels for message in extracted_messages])
        return clauses, labels


class Corpus(object):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X: np.ndarray = X
        self.Y: np.ndarray = Y
        self.grouped = group(self.X, self.Y, as_array=True)
        self.label_mapper = LabelMapper(Y=self.Y)

    @staticmethod
    def fromJSON(path: str, normalize_x: bool = False) -> "Corpus":
        X, Y = get_XY_json(path=path)
        if normalize_x:
            X = [normalize(x) for x in X]
        return Corpus(np.array(X), np.array(Y))

    @staticmethod
    def fromCSV(path: str, normalize_x: bool = False, text_kw: str = None, label_kw: str = None, text_index: int = None,
                label_index: int = None) -> "Corpus":

        df = pd.read_csv(path)
        X = df[text_kw] if text_kw is not None else df[:, text_index]
        Y = df[label_kw] if label_kw is not None else df[:, label_index]
        if normalize_x:
            X = [normalize(x) for x in X]
        return Corpus(np.array(X), np.array(Y))

    def reduce_by_label_size(self, min_label_size: int = 10) -> "Corpus":
        X_tmp, Y_tmp = [], []
        for x, y in zip(self.X, self.Y):
            if len(self.grouped[y]) >= min_label_size:
                X_tmp.append(x)
                Y_tmp.append(y)
        return Corpus(np.array(X_tmp), np.array(Y_tmp))

    def filter_for_labels(self, labels: Set[Union[str, int]], junk_label: Label = None) -> "Corpus":
        X_tmp, Y_tmp = [], []
        for x, y in zip(self.X, self.Y):
            if y in labels:
                X_tmp.append(x)
                Y_tmp.append(y)
            elif junk_label is not None:
                X_tmp.append(x)
                Y_tmp.append(junk_label)
        return Corpus(np.array(X_tmp), np.array(Y_tmp))

    def get_mapped_labels(self, datapoint_indices: np.ndarray = None) -> np.ndarray:
        datapoint_indices = np.arange(self.Y.shape[0]) if datapoint_indices is None else datapoint_indices
        return self.label_mapper.map_to_indices(self.Y[datapoint_indices])

    def get_data(self, datapoint_indices: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        datapoint_indices = np.arange(self.Y.shape[0]) if datapoint_indices is None else datapoint_indices
        return self.X[datapoint_indices], self.get_mapped_labels(datapoint_indices=datapoint_indices)

    def to_dict(self, x_name: str, y_name: str) -> Dict[str, np.ndarray]:
        return {
            x_name: self.X,
            y_name: self.Y
        }

    def __getitem__(self, item: slice):
        return Corpus(X=self.X[item], Y=self.Y[item])

    def shuffle(self) -> "Corpus":
        indices = np.random.choice(len(self.X), len(self.X), replace=False)
        return Corpus(X=self.X[indices], Y=self.Y[indices])

    def map_label(self, f: Callable[[Label], Label]) -> "Corpus":
        return Corpus(X=self.X, Y=np.array([f(y) for y in self.Y]))

    def add_entry(self, x: object, y: Label) -> "Corpus":
        return Corpus(X=np.hstack((self.X, [x])),
                      Y=np.hstack((self.Y, [y])))
