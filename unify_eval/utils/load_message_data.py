import gzip
import json
import os
from typing import Union, List, Dict, Iterable

from regex import regex

from unify_eval.utils.iter_utils import IterableIterator


def load_json(path: str) -> Union[str, list, dict, int, float, None]:
    with open(path, "r") as f:
        data = json.load(f)
    return data


def load_gzipped_json(path: str) -> Union[str, list, dict, int, float, None]:
    with gzip.open(path, "rb") as f:
        string = "".join([line.decode("utf-8") for line in f])
    return json.loads(string)


def load_data_file(path: str) -> Union[str, list, dict, int, float, None]:
    return load_gzipped_json(path=path) if path.endswith(".gz") else load_gzipped_json(path=path) if path.endswith(
        ".json") else {"messages": []}


class Message(object):
    def __init__(self, clauses: List[str]):
        self.clauses = clauses

    @staticmethod
    def from_dict(data: Dict[str, List[str]]) -> "Message":
        return Message([clause for clause in data["clauses"]])


def write_lines(lines: Iterable[str], path: str):
    with open(path, "w") as f:
        for i, line in enumerate(lines):
            f.write(line)
            f.write("\n")
        print(f"{i} lines written")


class MessageCorpusBatch(object):
    def __init__(self, messages: List[Message]):
        self.messages = messages

    def yield_messages(self) -> Iterable[Message]:
        return (message for message in self.messages)

    @staticmethod
    def from_file(path: str) -> "MessageCorpusBatch":
        return MessageCorpusBatch([Message(message["clauses"]) for message in load_data_file(path=path)["messages"]])


def yield_batched_texts(folder: str) -> Iterable[str]:
    for file in os.listdir(folder):
        if file.endswith(".json.gz"):
            file_path = os.path.join(folder, file)
            for entry in load_gzipped_json(path=file_path)["entries"]:
                yield entry


class MessageCorpus(object):
    def __init__(self, corpus_batches: Iterable[MessageCorpusBatch]):
        self.corpus_batches = corpus_batches

    def yield_messages(self) -> Iterable[Message]:
        return (message for corpus_batch in self.corpus_batches for message in corpus_batch.yield_messages())

    @staticmethod
    def from_folder(path: str) -> "MessageCorpus":
        files = os.listdir(path)

        return MessageCorpus(MessageCorpusBatch.from_file(os.path.join(path, file)) for file in files)


def yield_normalized_batched_texts(path_corpora: str) -> Iterable[str]:
    def normalize(s: str) -> str:
        s = s.lower()
        s = regex.sub(r"\s+", " ", s)
        s = regex.sub(r"ä", "ae", s)
        s = regex.sub(r"ö", "oe", s)
        s = regex.sub(r"ü", "ue", s)
        s = regex.sub(r"ß", "ss", s)
        return s

    def tokenize(s: str) -> List[str]:
        return s.split()

    preprocessed_corpus = IterableIterator(
        (" ".join(tokenize(normalize(entry))) for entry in yield_batched_texts(path_corpora)))
    return preprocessed_corpus
