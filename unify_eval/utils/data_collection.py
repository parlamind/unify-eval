import json
import os
from datetime import datetime
from typing import List, Iterable


class EvaluationDataConfig:
    """
    Class containing all evaluation types (train, test, validation ...) as well as metrics (accuracy,f1, model-dependent stuff)
    """

    def __init__(self,
                 evaluation_types: Iterable[str],
                 metrics: Iterable[str]):
        self.evaluation_types = evaluation_types
        self.metrics = metrics


class DataCollector:
    """
    Class collecting data as a list of dictionaries. Write everything into a json with a given path
    """

    def __init__(self, data: List[dict] = None):
        self.data = data if data is not None else []

    def add_data(self, **kwargs) -> "DataCollector":
        if "time" not in kwargs:
            kwargs["time"] = str(datetime.now())
        self.data.append(kwargs)
        return self

    def to_json(self, path: str):
        folder = os.path.join(*os.path.split(path)[:-1])
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(path, "w") as f:
            json.dump(self.data, f, indent=2)


class DataManager:
    """
    Class that maintains all DataCollectors of a given evaluation.
    """

    def __init__(self,
                 evaluation_data_config: EvaluationDataConfig,
                 path_to_folder: str):
        self.evaluation_data_config = evaluation_data_config
        self.data_collectors = dict((evaluation_type, DataCollector())
                                    for evaluation_type in self.evaluation_data_config.evaluation_types)
        self.path_to_folder = path_to_folder
        for evaluation_type in self.data_collectors:
            path_to_evaluation_type_folder = os.path.join(self.path_to_folder, evaluation_type)
            if not os.path.exists(path=path_to_evaluation_type_folder):
                print(f"writing evaluation data folder {path_to_evaluation_type_folder}")
                os.makedirs(path_to_evaluation_type_folder)

    def add_data(self, evaluation_type: str, global_step: int, **kwargs):
        self.data_collectors[evaluation_type].add_data(global_step=global_step, **kwargs)

    def write(self):
        for evaluation_type, collector in self.data_collectors.items():
            collector.to_json(path=os.path.join(self.path_to_folder, evaluation_type, "evaluation_data.json"))
