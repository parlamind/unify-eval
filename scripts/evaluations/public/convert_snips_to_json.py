import json
import os
import codecs
import regex
import numpy as np
from typing import Tuple, List, Dict

from unify_eval.utils.corpus import Corpus
from scripts.evaluations.public import data_utils


"""
This script was used to convert the SNIPS corpus to the JSON format used in Unifyeval. To reproduce the result:
1. download or clone the SNIPS repo from https://github.com/snipsco/nlu-benchmark
2. make sure the global variable SOURCE below points to the location where the SNIPS repo was downloaded or cloned
3. run this script, and the produced JSON files shall appear inside the data/public/snips/ folder
"""


SOURCE = "/home/mohamed/repos/nlu-benchmark/2017-06-custom-intent-engines"


def generate_corpus_from_folder(folder_path: str, full_data: bool = True, test_data:bool=False) -> Corpus:
    def merge_data_point(label: str, entry: dict) -> Tuple[str, str]:
        return "".join(token["text"] for token in entry["data"]), label

    def transform_label_data(label: str, label_data: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
        texts, labels = [], []
        for entry in label_data:
            text, label = merge_data_point(label, entry)
            texts.append(text)
            labels.append(label)
        return np.array(texts), np.array(labels)

    def transform_corpus(corpus: Dict[str, list]) -> Tuple[np.ndarray, np.ndarray]:
        texts, labels = [], []
        for label_name, label_data in corpus.items():
            for text, label in zip(*transform_label_data(label_name, label_data)):
                texts.append(text)
                labels.append(label)
        return np.array(texts), np.array(labels)

    def generate_corpus(*single_file_paths: str) -> Tuple[np.ndarray, np.ndarray]:
        texts, labels = [], []
        for path in single_file_paths:

            with codecs.open(path, 'r', encoding='utf-8', errors='ignore') as f:
                raw_training_corpus = json.load(f)

            for text, label in zip(*transform_corpus(raw_training_corpus)):
                texts.append(text)
                labels.append(label)
        return np.array(texts), np.array(labels)

    texts, labels = [], []
    prefix = "validate" if test_data else "train"
    small_pattern = rf"{prefix}_[a-zA-Z]+\.json"
    full_pattern = rf"{prefix}_[a-zA-Z]+_full\.json" if not test_data else small_pattern
    for file_name in os.listdir(folder_path):
        full_path = os.path.join(folder_path, file_name)
        if os.path.isdir(full_path):
            for json_file in os.listdir(full_path):
                if regex.match(pattern=full_pattern if full_data else small_pattern,
                               string=json_file):
                    full_file_path = os.path.join(full_path, json_file)
                    print(f"reading file {full_file_path}")
                    for text, label in zip(*generate_corpus(full_file_path)):
                        texts.append(text)
                        labels.append(label)
    return Corpus(np.array(texts), np.array(labels))


def generate_message_corpus_from_folder(folder_path: str,
                                        full_data: bool,
                                        test_data: bool) -> List[dict]:
    corpus = generate_corpus_from_folder(folder_path=folder_path,
                                         full_data=full_data,
                                         test_data=test_data)

    return data_utils.rewrite_corpus(corpus)


def generate_regular_corpus_from_folder(folder_path: str,
                                        full_data: bool,
                                        test_data: bool) -> List[dict]:
    corpus = generate_corpus_from_folder(folder_path=folder_path,
                                         full_data=full_data,
                                         test_data=test_data)

    return data_utils.rewrite_corpus(corpus, regular=True)


def convert_and_dump_to_json(test_data: bool, regular_format: bool):
    name = "test" if test_data else "train"
    if not regular_format:
        name = name + "_isolated"
    target = os.path.join("data", "public", "snips", name+".json")

    if regular_format:
        corpus = generate_regular_corpus_from_folder(SOURCE, full_data=True, test_data=test_data)
    else:
        corpus = generate_message_corpus_from_folder(SOURCE, True, test_data=test_data)

    with open(target, "w") as f:
        json.dump(corpus, f, indent=2)


def main():
    convert_and_dump_to_json(test_data=False, regular_format=True)
    convert_and_dump_to_json(test_data=True, regular_format=True)
    convert_and_dump_to_json(test_data=True, regular_format=False)


if __name__ == "__main__":
    main()
