import os
import json
from typing import Tuple
import numpy as np

from unify_eval.model.types import Label
from unify_eval.utils.corpus import Corpus
from scripts.evaluations.public import data_utils


"""
This script was used to convert the Sebischair corpus to the JSON format used in Unifyeval. To reproduce the result:
1. download or clone the Sebischair repo from https://github.com/sebischair/NLU-Evaluation-Corpora
2. make sure the global variable REPO below points to the location where the Sebischair repo was downloaded or cloned
3. run this script, and the produced JSON files shall appear inside the data/public/sebischair/ folder
"""

REPO = "/home/mohamed/repos/NLU-Evaluation-Corpora/"


def generate_corpus_from_file(path: str) -> Tuple[Corpus, Corpus]:
    def transform_entry(entry: dict) -> Tuple[str, Label, bool]:
        return entry["text"], entry["intent"], entry["training"]

    def transform_corpus(corpus: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        texts, labels, is_trainings = [], [], []
        for entry in corpus["sentences"]:
            text, label, is_training = transform_entry(entry)
            texts.append(text)
            labels.append(label)
            is_trainings.append(is_training)
        return np.array(texts), np.array(labels), np.array(is_trainings)

    def generate_corpus(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        with open(path, "r") as f:
            raw_training_corpus = json.load(f)

        return transform_corpus(raw_training_corpus)

    texts, labels, is_training = generate_corpus(path=path)
    return Corpus(texts[is_training], labels[is_training]), \
           Corpus(texts[np.bitwise_not(is_training)], labels[np.bitwise_not(is_training)])


def write_corpus_to_file(corpus, name, testing, regular):
    corpus_as_dict = data_utils.rewrite_corpus(corpus, regular=regular)
    complete_name = name + "_test" if testing else name + "_train"
    if not regular:
        complete_name = complete_name + "_isolated"
    complete_name = complete_name + ".json"
    target = os.path.join("data", "public", "sebischair", complete_name)

    with open(target, "w") as f:
        json.dump(corpus_as_dict, f, indent=2)


def convert_and_store_corpora(name):
    path = REPO + name + ".json"
    train_corpus, test_corpus = generate_corpus_from_file(path)
    write_corpus_to_file(train_corpus, name, testing=False, regular=True)
    write_corpus_to_file(test_corpus, name, testing=True, regular=True)
    write_corpus_to_file(test_corpus, name, testing=True, regular=False)


def main():
    corpora_names = ["AskUbuntuCorpus", "ChatbotCorpus", "WebApplicationsCorpus"]
    for corpus_name in corpora_names:
        convert_and_store_corpora(corpus_name)


if __name__ == "__main__":
    main()
